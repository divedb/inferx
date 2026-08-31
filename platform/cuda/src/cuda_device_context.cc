#include "inferx/platform/cuda/cuda_device_context.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"
#include "inferx/platform/cuda/cuda_error.h"

namespace inferx::cuda {
namespace {

absl::StatusOr<uint64_t> PoolBytes(uint32_t slots, ByteCount bytes, absl::string_view context) {
  return CheckedMul(static_cast<uint64_t>(slots), bytes.value(), context);
}

absl::Status AnnotateCleanup(absl::Status primary, const absl::Status& cleanup) {
  if (cleanup.ok()) return primary;
  absl::Status annotated(
      primary.code(), absl::StrCat(primary.message(), "; cleanup failure: ", cleanup.ToString()));
  primary.ForEachPayload([&annotated](absl::string_view url, const absl::Cord& payload) {
    annotated.SetPayload(url, payload);
  });
  return annotated;
}

void AccumulateCleanup(absl::Status candidate, absl::Status* aggregate) {
  if (candidate.ok()) return;
  if (aggregate->ok()) {
    *aggregate = std::move(candidate);
    return;
  }
  *aggregate = absl::Status(
      aggregate->code(),
      absl::StrCat(aggregate->message(), "; additional cleanup failure: ", candidate.ToString()));
}

absl::Status ValidateContextGeometry(const CudaContextConfig& config) {
  constexpr uint64_t kMiB = 1024ULL * 1024;
  constexpr uint64_t kGiB = 1024ULL * kMiB;
  if (config.device.value() > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return absl::InvalidArgumentError("cuda.context.device: ordinal cannot narrow to int");
  }
  if (config.accepted_sm <= 0) {
    return absl::InvalidArgumentError("cuda.context.accepted_sm: value must be positive");
  }
  if (config.pinned_budget.value() < kMiB || config.pinned_budget.value() > 4 * kGiB) {
    return absl::InvalidArgumentError(
        "cuda.context.pinned_budget_bytes: value must be in [1 MiB, 4 GiB]");
  }
  if (config.event_pool_slots < 8 || config.event_pool_slots > 65536) {
    return absl::InvalidArgumentError("cuda.context.event_pool_slots: value must be in [8, 65536]");
  }
  if (config.metadata_ring_slots < 2 || config.metadata_ring_slots > 64 ||
      config.metadata_slot_bytes.value() < 4096 || config.metadata_slot_bytes.value() > 16 * kMiB ||
      !std::has_single_bit(config.metadata_slot_bytes.value())) {
    return absl::InvalidArgumentError(
        "cuda.context.metadata_geometry: slots or slot bytes are outside the supported range");
  }
  if (config.staging_pool_slots < 2 || config.staging_pool_slots > 256 ||
      (config.staging_pool_slots & 1U) != 0 || config.staging_slot_bytes.value() < 4096 ||
      config.staging_slot_bytes.value() > 64 * kMiB ||
      !std::has_single_bit(config.staging_slot_bytes.value())) {
    return absl::InvalidArgumentError(
        "cuda.context.staging_geometry: slots or slot bytes are outside the supported range");
  }
  if (config.workspace_slots == 0 || config.workspace_slots > 64 ||
      config.workspace_bytes_per_slot.value() < kMiB ||
      config.workspace_bytes_per_slot.value() > 4 * kGiB) {
    return absl::InvalidArgumentError(
        "cuda.context.workspace_geometry: slots or slot bytes are outside the supported range");
  }
  return absl::OkStatus();
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
  std::vector<std::unique_ptr<CudaDeferredResource>> deferred_resources;
  bool synchronized = false;
  bool closed = false;

  absl::Status CleanupConstructedPrefix() {
    absl::Status cleanup = absl::OkStatus();
    if (workspace_pool.has_value()) {
      AccumulateCleanup(workspace_pool->Close(), &cleanup);
    }
    if (metadata_ring.has_value()) {
      AccumulateCleanup(metadata_ring->Close(), &cleanup);
    }
    if (staging_pool.has_value()) {
      AccumulateCleanup(staging_pool->Close(), &cleanup);
    }
    if (event_pool != nullptr) {
      AccumulateCleanup(event_pool->Close(), &cleanup);
    }
    if (transfer_stream.has_value()) {
      AccumulateCleanup(transfer_stream->Close(), &cleanup);
    }
    if (compute_stream.has_value()) {
      AccumulateCleanup(compute_stream->Close(), &cleanup);
    }
    device_allocator.reset();
    pinned_allocator.reset();
    if (tracker != nullptr) {
      AccumulateCleanup(tracker->ValidateBaseline(), &cleanup);
    }
    return cleanup;
  }

  absl::Status CreationFailure(absl::Status primary) {
    return AnnotateCleanup(std::move(primary), CleanupConstructedPrefix());
  }
};

CudaDeviceContext::CudaDeviceContext(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

absl::StatusOr<std::unique_ptr<CudaDeviceContext>> CudaDeviceContext::Create(
    const CudaContextConfig& config, MemoryTracker& tracker, const CudaApi& api) {
  absl::Status geometry = ValidateContextGeometry(config);
  if (!geometry.ok()) return geometry;
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
  std::unique_ptr<Impl> impl;
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

    impl = std::make_unique<Impl>();
    impl->api = &api;
    impl->tracker = &tracker;
    impl->config = config;
    impl->info = *selected;
    impl->deferred_resources.reserve(static_cast<size_t>(max_test_inflight));
    absl::Status set_device =
        CudaErrorStatus(api.set_device(static_cast<int>(config.device.value())), "set-device",
                        config.device, &impl->health);
    if (!set_device.ok()) return impl->CreationFailure(set_device);
    impl->device_allocator =
        std::make_unique<CudaDeviceAllocator>(config.device, tracker, impl->health, api);
    impl->pinned_allocator =
        std::make_unique<CudaPinnedAllocator>(config.device, tracker, impl->health, api);
    absl::StatusOr<CudaStream> compute =
        CudaStream::Create(config.device, CudaStreamRole::kCompute, 0, api, &impl->health);
    if (!compute.ok()) return impl->CreationFailure(compute.status());
    impl->compute_stream.emplace(std::move(*compute));
    if (config.enable_transfer_stream) {
      absl::StatusOr<CudaStream> transfer =
          CudaStream::Create(config.device, CudaStreamRole::kTransfer, 0, api, &impl->health);
      if (!transfer.ok()) return impl->CreationFailure(transfer.status());
      impl->transfer_stream.emplace(std::move(*transfer));
    }
    absl::StatusOr<std::unique_ptr<CudaEventPool>> events =
        CudaEventPool::Create(config.device, config.event_pool_slots, api, &impl->health);
    if (!events.ok()) return impl->CreationFailure(events.status());
    impl->event_pool = std::move(*events);

    const AllocationRequest staging_request{Device::Host(), MemoryKind::kPinnedHost,
                                            ByteCount(*staging_bytes), ByteCount(256),
                                            MemoryCategory::kPinnedStaging};
    absl::StatusOr<Buffer> staging_backing = impl->pinned_allocator->Allocate(staging_request);
    if (!staging_backing.ok()) return impl->CreationFailure(staging_backing.status());
    absl::StatusOr<FixedBufferPool> staging =
        FixedBufferPool::Create(std::move(*staging_backing),
                                PoolGeometry{config.staging_pool_slots, config.staging_slot_bytes,
                                             ByteCount(256), PoolGeneration(0)});
    if (!staging.ok()) return impl->CreationFailure(staging.status());
    impl->staging_pool.emplace(std::move(*staging));

    const AllocationRequest workspace_request{Device::Cuda(config.device), MemoryKind::kDevice,
                                              ByteCount(*workspace_bytes), ByteCount(256),
                                              MemoryCategory::kWorkspace};
    absl::StatusOr<Buffer> workspace_backing = impl->device_allocator->Allocate(workspace_request);
    if (!workspace_backing.ok()) return impl->CreationFailure(workspace_backing.status());
    absl::StatusOr<WorkspaceArenaPool> workspace =
        WorkspaceArenaPool::Create(std::move(*workspace_backing), config.workspace_slots,
                                   config.workspace_bytes_per_slot, ByteCount(256));
    if (!workspace.ok()) return impl->CreationFailure(workspace.status());
    impl->workspace_pool.emplace(std::move(*workspace));

    absl::StatusOr<CudaMetadataRing> metadata = CudaMetadataRing::Create(
        config.metadata_ring_slots, config.metadata_slot_bytes, *impl->pinned_allocator,
        *impl->device_allocator, *impl->event_pool, config.device, api, &impl->health);
    if (!metadata.ok()) return impl->CreationFailure(metadata.status());
    impl->metadata_ring.emplace(std::move(*metadata));
    return std::unique_ptr<CudaDeviceContext>(new CudaDeviceContext(std::move(impl)));
  } catch (const std::bad_alloc&) {
    absl::Status failure =
        absl::ResourceExhaustedError("cuda.context: host resource construction failed");
    return impl == nullptr ? failure : impl->CreationFailure(std::move(failure));
  }
}

CudaDeviceContext::~CudaDeviceContext() noexcept {
  if (impl_ != nullptr && !impl_->closed) {
    std::terminate();
  }
}

const CudaDeviceInfo& CudaDeviceContext::info() const noexcept { return impl_->info; }
const CudaApi& CudaDeviceContext::api() const noexcept { return *impl_->api; }
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

absl::Status CudaDeviceContext::DeferResource(std::unique_ptr<CudaDeferredResource> resource) {
  if (resource == nullptr) {
    return absl::InvalidArgumentError("cuda.context.defer: resource is null");
  }
  if (impl_->closed) {
    static_cast<void>(resource.release());
    return absl::FailedPreconditionError("cuda.context.defer: context is closed");
  }
  if (impl_->deferred_resources.size() == impl_->deferred_resources.capacity()) {
    static_cast<void>(resource.release());
    absl::Status failure = WithErrorReason(
        absl::InternalError("cuda.context.defer: reserved deferred capacity exhausted"),
        ErrorReason::kInvariantViolation);
    impl_->health.Poison(failure);
    return failure;
  }
  try {
    impl_->deferred_resources.push_back(std::move(resource));
  } catch (const std::bad_alloc&) {
    static_cast<void>(resource.release());
    absl::Status failure = WithErrorReason(
        absl::ResourceExhaustedError("cuda.context.defer: host ownership transfer failed"),
        ErrorReason::kInvariantViolation);
    impl_->health.Poison(failure);
    return failure;
  }
  return absl::OkStatus();
}

absl::Status CudaDeviceContext::ReclaimDeferredResources() {
  absl::Status first_error = absl::OkStatus();
  for (std::unique_ptr<CudaDeferredResource>& resource : impl_->deferred_resources) {
    absl::StatusOr<bool> reclaimed = resource->TryReclaim();
    if (!reclaimed.ok()) {
      if (first_error.ok()) first_error = reclaimed.status();
      continue;
    }
    if (*reclaimed) resource.reset();
  }
  size_t destination = 0;
  for (size_t index = 0; index < impl_->deferred_resources.size(); ++index) {
    if (impl_->deferred_resources[index] != nullptr) {
      impl_->deferred_resources[destination++] = std::move(impl_->deferred_resources[index]);
    }
  }
  impl_->deferred_resources.resize(destination);
  return first_error;
}

size_t CudaDeviceContext::deferred_resource_count() const noexcept {
  return impl_->deferred_resources.size();
}

absl::Status CudaDeviceContext::Shutdown(Deadline deadline) {
  if (impl_->closed) return absl::OkStatus();
  absl::Status transition = impl_->health.BeginShutdown();
  if (!transition.ok()) return transition;
  while (impl_->deferred_resources.size() != 0 || impl_->event_pool->has_outstanding_events()) {
    absl::Status deferred = ReclaimDeferredResources();
    if (!deferred.ok()) return deferred;
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
  absl::Status workspace = impl_->workspace_pool->Close();
  if (!workspace.ok()) return workspace;
  absl::Status metadata = impl_->metadata_ring->Close();
  if (!metadata.ok()) return metadata;
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
