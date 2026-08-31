#include "inferx/platform/cuda/cuda_device_context.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"
#include "inferx/platform/cuda/cuda_error.h"

namespace inferx::cuda {
namespace {

absl::StatusOr<uint64_t> PoolBytes(uint32_t slots, ByteCount bytes, absl::string_view context) {
  return CheckedMul(static_cast<uint64_t>(slots), bytes.value(), context);
}

}  // namespace

struct CudaDeviceContext::Impl {
  const CudaApi* api = nullptr;
  MemoryTracker* tracker = nullptr;
  CudaContextConfig config;
  CudaHealth health;
  CudaDeviceInfo info{DeviceId(0), "",    0,     0,     ByteCount(0), 0,     0, 0,
                      0,           false, false, false, false,        false, 0, 0};
  std::unique_ptr<CudaDeviceAllocator> device_allocator;
  std::unique_ptr<CudaPinnedAllocator> pinned_allocator;
  std::optional<CudaStream> compute_stream;
  std::optional<CudaStream> transfer_stream;
  std::unique_ptr<CudaEventPool> event_pool;
  std::optional<FixedBufferPool> staging_pool;
  std::optional<CudaMetadataRing> metadata_ring;
  std::optional<WorkspaceArenaPool> workspace_pool;
  bool synchronized = false;
  bool closed = false;
};

CudaDeviceContext::CudaDeviceContext(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

absl::StatusOr<std::unique_ptr<CudaDeviceContext>> CudaDeviceContext::Create(
    const CudaContextConfig& config, MemoryTracker& tracker, const CudaApi& api) {
  if (config.metadata_ring_slots < 2 || config.staging_pool_slots < 2 ||
      (config.staging_pool_slots & 1U) != 0 || config.workspace_slots == 0 ||
      config.pinned_budget.value() == 0 || config.metadata_slot_bytes.value() < 4096 ||
      !std::has_single_bit(config.metadata_slot_bytes.value()) ||
      config.staging_slot_bytes.value() < 4096 ||
      !std::has_single_bit(config.staging_slot_bytes.value()) ||
      config.workspace_bytes_per_slot.value() == 0) {
    return absl::InvalidArgumentError(
        "cuda.context.geometry: invalid metadata, staging, or workspace geometry");
  }
  absl::StatusOr<uint64_t> staging_bytes =
      PoolBytes(config.staging_pool_slots, config.staging_slot_bytes, "cuda.context.staging_bytes");
  if (!staging_bytes.ok()) return staging_bytes.status();
  absl::StatusOr<uint64_t> metadata_bytes = PoolBytes(
      config.metadata_ring_slots, config.metadata_slot_bytes, "cuda.context.metadata_bytes");
  if (!metadata_bytes.ok()) return metadata_bytes.status();
  absl::StatusOr<uint64_t> workspace_bytes = PoolBytes(
      config.workspace_slots, config.workspace_bytes_per_slot, "cuda.context.workspace_bytes");
  if (!workspace_bytes.ok()) return workspace_bytes.status();
  absl::StatusOr<uint64_t> pinned_startup =
      CheckedAdd(*staging_bytes, *metadata_bytes, "cuda.context.pinned_budget_bytes");
  if (!pinned_startup.ok()) return pinned_startup.status();
  if (*pinned_startup > config.pinned_budget.value()) {
    return absl::InvalidArgumentError(
        "cuda.context.pinned_budget_bytes: startup pools exceed budget");
  }
  const uint64_t max_test_inflight =
      std::min({static_cast<uint64_t>(config.metadata_ring_slots),
                static_cast<uint64_t>(config.workspace_slots),
                static_cast<uint64_t>(config.staging_pool_slots / 2)});
  absl::StatusOr<uint64_t> completion_events =
      CheckedMul(max_test_inflight, uint64_t{2}, "cuda.context.event_pool_slots");
  if (!completion_events.ok()) return completion_events.status();
  absl::StatusOr<uint64_t> required_events =
      CheckedAdd(static_cast<uint64_t>(config.metadata_ring_slots), *completion_events,
                 "cuda.context.event_pool_slots");
  if (!required_events.ok()) return required_events.status();
  if (static_cast<uint64_t>(config.event_pool_slots) < *required_events) {
    return absl::InvalidArgumentError("cuda.context.event_pool_slots: insufficient event capacity");
  }
  try {
    absl::StatusOr<std::vector<CudaDeviceInfo>> devices = DiscoverCudaDevices(api);
    if (!devices.ok()) return devices.status();
    const auto selected =
        std::find_if(devices->begin(), devices->end(),
                     [&](const CudaDeviceInfo& info) { return info.ordinal == config.device; });
    if (selected == devices->end()) {
      return absl::NotFoundError("cuda.context.device: selected device does not exist");
    }
    const int accepted_sm = config.accepted_sm;
    absl::Status capability =
        ValidateCudaCapabilities(*selected, std::span<const int>(&accepted_sm, 1),
                                 config.device_reserve, config.device_budget);
    if (!capability.ok()) return capability;
    const uint64_t resolved_device_budget =
        config.device_budget.value() == 0
            ? selected->total_memory.value() - config.device_reserve.value()
            : config.device_budget.value();
    absl::StatusOr<uint64_t> device_startup =
        CheckedAdd(*workspace_bytes, *metadata_bytes, "cuda.context.device_budget_bytes");
    if (!device_startup.ok()) return device_startup.status();
    if (*device_startup > resolved_device_budget) {
      return absl::InvalidArgumentError(
          "cuda.context.device_budget_bytes: startup pools exceed budget");
    }

    auto impl = std::make_unique<Impl>();
    impl->api = &api;
    impl->tracker = &tracker;
    impl->config = config;
    impl->info = *selected;
    absl::Status set_device =
        CudaErrorStatus(api.set_device(static_cast<int>(config.device.value())), "set-device",
                        config.device, &impl->health);
    if (!set_device.ok()) return set_device;
    impl->device_allocator =
        std::make_unique<CudaDeviceAllocator>(config.device, tracker, impl->health, api);
    impl->pinned_allocator =
        std::make_unique<CudaPinnedAllocator>(config.device, tracker, impl->health, api);
    absl::StatusOr<CudaStream> compute =
        CudaStream::Create(config.device, CudaStreamRole::kCompute, 0, api, &impl->health);
    if (!compute.ok()) return compute.status();
    impl->compute_stream.emplace(std::move(*compute));
    if (config.enable_transfer_stream) {
      absl::StatusOr<CudaStream> transfer =
          CudaStream::Create(config.device, CudaStreamRole::kTransfer, 0, api, &impl->health);
      if (!transfer.ok()) return transfer.status();
      impl->transfer_stream.emplace(std::move(*transfer));
    }
    absl::StatusOr<std::unique_ptr<CudaEventPool>> events =
        CudaEventPool::Create(config.device, config.event_pool_slots, api, &impl->health);
    if (!events.ok()) return events.status();
    impl->event_pool = std::move(*events);

    const AllocationRequest staging_request{Device::Host(), MemoryKind::kPinnedHost,
                                            ByteCount(*staging_bytes), ByteCount(256),
                                            MemoryCategory::kPinnedStaging};
    absl::StatusOr<Buffer> staging_backing = impl->pinned_allocator->Allocate(staging_request);
    if (!staging_backing.ok()) return staging_backing.status();
    absl::StatusOr<FixedBufferPool> staging =
        FixedBufferPool::Create(std::move(*staging_backing),
                                PoolGeometry{config.staging_pool_slots, config.staging_slot_bytes,
                                             ByteCount(256), PoolGeneration(0)});
    if (!staging.ok()) return staging.status();
    impl->staging_pool.emplace(std::move(*staging));

    const AllocationRequest workspace_request{Device::Cuda(config.device), MemoryKind::kDevice,
                                              ByteCount(*workspace_bytes), ByteCount(256),
                                              MemoryCategory::kWorkspace};
    absl::StatusOr<Buffer> workspace_backing = impl->device_allocator->Allocate(workspace_request);
    if (!workspace_backing.ok()) return workspace_backing.status();
    absl::StatusOr<WorkspaceArenaPool> workspace =
        WorkspaceArenaPool::Create(std::move(*workspace_backing), config.workspace_slots,
                                   config.workspace_bytes_per_slot, ByteCount(256));
    if (!workspace.ok()) return workspace.status();
    impl->workspace_pool.emplace(std::move(*workspace));

    absl::StatusOr<CudaMetadataRing> metadata = CudaMetadataRing::Create(
        config.metadata_ring_slots, config.metadata_slot_bytes, *impl->pinned_allocator,
        *impl->device_allocator, *impl->event_pool, config.device, api);
    if (!metadata.ok()) return metadata.status();
    impl->metadata_ring.emplace(std::move(*metadata));
    return std::unique_ptr<CudaDeviceContext>(new CudaDeviceContext(std::move(impl)));
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("cuda.context: host resource construction failed");
  }
}

CudaDeviceContext::~CudaDeviceContext() noexcept = default;

const CudaDeviceInfo& CudaDeviceContext::info() const noexcept { return impl_->info; }
CudaHealth& CudaDeviceContext::health() noexcept { return impl_->health; }
CudaStream& CudaDeviceContext::compute_stream() noexcept { return *impl_->compute_stream; }
CudaStream& CudaDeviceContext::transfer_stream() noexcept {
  return impl_->transfer_stream.has_value() ? *impl_->transfer_stream : *impl_->compute_stream;
}
CudaEventPool& CudaDeviceContext::event_pool() noexcept { return *impl_->event_pool; }
CudaDeviceAllocator& CudaDeviceContext::device_allocator() noexcept {
  return *impl_->device_allocator;
}
CudaPinnedAllocator& CudaDeviceContext::pinned_allocator() noexcept {
  return *impl_->pinned_allocator;
}
FixedBufferPool& CudaDeviceContext::staging_pool() noexcept { return *impl_->staging_pool; }
WorkspaceArenaPool& CudaDeviceContext::workspace_pool() noexcept { return *impl_->workspace_pool; }
CudaMetadataRing& CudaDeviceContext::metadata_ring() noexcept { return *impl_->metadata_ring; }

absl::Status CudaDeviceContext::Shutdown(Deadline deadline) {
  if (impl_->closed) return absl::OkStatus();
  absl::Status transition = impl_->health.BeginShutdown();
  if (!transition.ok()) return transition;
  while (impl_->event_pool->available_slots() != impl_->config.event_pool_slots) {
    absl::Status reclaim = impl_->event_pool->ReclaimAbandoned();
    if (!reclaim.ok()) return reclaim;
    const MonotonicTime now =
        std::chrono::time_point_cast<Nanoseconds>(std::chrono::steady_clock::now());
    if (now >= deadline) {
      return absl::DeadlineExceededError(
          "cuda.context.shutdown: completion drain deadline expired");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  if (!impl_->synchronized) {
    absl::Status sync =
        CudaErrorStatus(impl_->api->device_synchronize(), "shutdown-device-synchronize",
                        impl_->config.device, &impl_->health);
    if (!sync.ok()) return sync;
    impl_->synchronized = true;
  }
  absl::Status metadata = impl_->metadata_ring->Close();
  if (!metadata.ok()) return metadata;
  absl::Status workspace = impl_->workspace_pool->Close();
  if (!workspace.ok()) return workspace;
  absl::Status staging = impl_->staging_pool->Close();
  if (!staging.ok()) return staging;
  absl::Status events = impl_->event_pool->Close();
  if (!events.ok()) return events;
  if (impl_->transfer_stream.has_value()) {
    absl::Status transfer = impl_->transfer_stream->Close();
    if (!transfer.ok()) return transfer;
  }
  absl::Status compute = impl_->compute_stream->Close();
  if (!compute.ok()) return compute;
  absl::Status baseline = impl_->tracker->ValidateBaseline();
  if (!baseline.ok()) return baseline;
  absl::Status closed = impl_->health.Close();
  if (!closed.ok()) return closed;
  impl_->closed = true;
  return absl::OkStatus();
}

}  // namespace inferx::cuda
