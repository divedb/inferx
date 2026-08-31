#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "inferx/base/clock.h"
#include "inferx/platform/cuda/cuda_allocator.h"
#include "inferx/platform/cuda/cuda_api.h"
#include "inferx/platform/cuda/cuda_buffer_access.h"
#include "inferx/platform/cuda/cuda_copy.h"
#include "inferx/platform/cuda/cuda_device.h"
#include "inferx/platform/cuda/cuda_device_context.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/cuda_event_pool.h"
#include "inferx/platform/cuda/cuda_stream.h"
#include "inferx/runtime/memory_tracker.h"
#include "inferx/tensor/buffer.h"
#include "test_kernels.h"

namespace {

using inferx::Buffer;
using inferx::BufferLease;
using inferx::BufferView;
using inferx::ByteCount;
using inferx::CompletionFence;
using inferx::CpuAllocator;
using inferx::Device;
using inferx::DeviceId;
using inferx::FencePoll;
using inferx::FenceState;
using inferx::FenceWaitReason;
using inferx::MemoryCategory;
using inferx::MemoryKind;
using inferx::MemoryLimit;
using inferx::MemoryTracker;
using inferx::MutableBufferView;
using inferx::Nanoseconds;
using inferx::WorkspaceLease;
using inferx::cuda::BufferAccess;
using inferx::cuda::CopyAsync;
using inferx::cuda::CopyRequest;
using inferx::cuda::CudaContextConfig;
using inferx::cuda::CudaDeviceAllocator;
using inferx::cuda::CudaDeviceContext;
using inferx::cuda::CudaDeviceGuard;
using inferx::cuda::CudaDeviceInfo;
using inferx::cuda::CudaErrorStatus;
using inferx::cuda::CudaEventLease;
using inferx::cuda::CudaEventPool;
using inferx::cuda::CudaHealth;
using inferx::cuda::CudaPinnedAllocator;
using inferx::cuda::CudaStream;
using inferx::cuda::CudaStreamRole;

constexpr int kAcceptedSm = 89;
constexpr uint64_t kTestBytes = 4096;
constexpr uint64_t kPinnedBudget = 32ULL * 1024 * 1024;
constexpr uint64_t kDeviceBudget = 32ULL * 1024 * 1024;

inferx::Deadline DeadlineAfter(std::chrono::seconds duration) {
  return std::chrono::time_point_cast<Nanoseconds>(std::chrono::steady_clock::now()) + duration;
}

CudaContextConfig TestContextConfig(DeviceId device) {
  CudaContextConfig config;
  config.device = device;
  config.accepted_sm = kAcceptedSm;
  config.device_reserve = ByteCount(0);
  config.device_budget = ByteCount(kDeviceBudget);
  config.pinned_budget = ByteCount(kPinnedBudget);
  config.event_pool_slots = 8;
  config.metadata_ring_slots = 2;
  config.metadata_slot_bytes = ByteCount(kTestBytes);
  config.staging_pool_slots = 2;
  config.staging_slot_bytes = ByteCount(kTestBytes);
  config.workspace_slots = 1;
  config.workspace_bytes_per_slot = ByteCount(1024 * 1024);
  config.enable_transfer_stream = true;
  return config;
}

absl::Status SelfTest(const CudaDeviceInfo& info) {
  const auto& api = inferx::cuda::CudaApi::Production();
  const MemoryLimit limits[] = {
      {Device::Host(), MemoryKind::kPinnedHost, ByteCount(kPinnedBudget)},
      {Device::Cuda(info.ordinal), MemoryKind::kDevice, ByteCount(kDeviceBudget)}};
  absl::StatusOr<MemoryTracker> tracker = MemoryTracker::Create(limits);
  if (!tracker.ok()) return tracker.status();
  absl::StatusOr<std::unique_ptr<CudaDeviceContext>> created =
      CudaDeviceContext::Create(TestContextConfig(info.ordinal), *tracker, api);
  if (!created.ok()) return created.status();
  std::unique_ptr<CudaDeviceContext> context = std::move(*created);

  absl::StatusOr<BufferLease> input = context->staging_pool().Acquire();
  if (!input.ok()) return input.status();
  absl::StatusOr<BufferLease> output = context->staging_pool().Acquire();
  if (!output.ok()) return output.status();
  absl::StatusOr<inferx::cuda::CudaMetadataLease> metadata = context->metadata_ring().Acquire();
  if (!metadata.ok()) return metadata.status();
  absl::StatusOr<WorkspaceLease> workspace = context->workspace_pool().Acquire();
  if (!workspace.ok()) return workspace.status();
  absl::StatusOr<MutableBufferView> device_input =
      workspace->Allocate(ByteCount(kTestBytes), ByteCount(256), inferx::WorkspaceTag::kTest);
  if (!device_input.ok()) return device_input.status();
  absl::StatusOr<MutableBufferView> device_output =
      workspace->Allocate(ByteCount(kTestBytes), ByteCount(256), inferx::WorkspaceTag::kTest);
  if (!device_output.ok()) return device_output.status();
  absl::StatusOr<MutableBufferView> scratch =
      workspace->Allocate(ByteCount(256), ByteCount(64), inferx::WorkspaceTag::kTest);
  if (!scratch.ok()) return scratch.status();

  MutableBufferView input_view = input->mutable_view().value();
  MutableBufferView output_view = output->mutable_view().value();
  for (uint64_t index = 0; index < kTestBytes; ++index) {
    (*input_view.HostBytes())[index] = static_cast<std::byte>((index * 31) & 0xff);
  }
  std::memset(output_view.HostBytes()->data(), 0, output_view.HostBytes()->size());
  MutableBufferView metadata_host = metadata->host_view().value();
  std::memset(metadata_host.HostBytes()->data(), 0x5a, metadata_host.HostBytes()->size());

  absl::Status copy =
      CopyAsync(CopyRequest{input_view.AsConst(), *device_input, ByteCount(kTestBytes)},
                context->transfer_stream(), api, &context->health());
  if (!copy.ok()) return copy;
  // Metadata upload is enqueued after input H2D on the same transfer stream;
  // its retained event therefore covers both transfers before compute starts.
  absl::StatusOr<BufferView> device_metadata =
      metadata->SealAndUpload(context->transfer_stream(), context->compute_stream());
  if (!device_metadata.ok()) return device_metadata.status();

  inferx::cuda::test::StridedCopyParams params;
  params.dimensions[0] = kTestBytes;
  params.source_strides[0] = 1;
  params.destination_strides[0] = 1;
  params.element_count = kTestBytes;
  params.element_size = 1;
  params.rank = 1;
  inferx::cuda::test::LaunchStridedCopy(BufferAccess::Address(device_input->AsConst()),
                                        BufferAccess::Address(*device_output), params,
                                        context->compute_stream().handle());
  // A small correct barrier-bearing fixture gives racecheck/synccheck a real
  // synchronization path. It is ordered after the copy and its output is not
  // part of the byte-copy result, so use the input allocation temporarily.
  inferx::cuda::test::LaunchSynchronizationProbe(
      static_cast<uint32_t*>(BufferAccess::Address(*device_input)),
      context->compute_stream().handle());
  // Restore the byte-copy result after the probe overwrote device_input only;
  // device_output, copied below, remains the correctness result.
  if (cudaError_t error = api.peek_at_last_error(); error != cudaSuccess) {
    return CudaErrorStatus(error, "strided-copy-launch", info.ordinal, &context->health());
  }
  absl::StatusOr<CudaEventLease> compute_done = context->event_pool().Acquire();
  if (!compute_done.ok()) return compute_done.status();
  if (absl::Status status = compute_done->Record(context->compute_stream()); !status.ok()) {
    return status;
  }
  if (absl::Status status = compute_done->WaitOn(context->transfer_stream()); !status.ok()) {
    return status;
  }
  copy = CopyAsync(CopyRequest{device_output->AsConst(), output_view, ByteCount(kTestBytes)},
                   context->transfer_stream(), api, &context->health());
  if (!copy.ok()) return copy;
  absl::StatusOr<CudaEventLease> completed = context->event_pool().Acquire();
  if (!completed.ok()) return completed.status();
  if (absl::Status status = completed->Record(context->transfer_stream()); !status.ok()) {
    return status;
  }
  absl::StatusOr<CompletionFence> fence = completed->IntoFence();
  if (!fence.ok()) return fence.status();
  const inferx::Deadline deadline = DeadlineAfter(std::chrono::seconds(5));
  if (absl::Status status = fence->WaitUntil(deadline, FenceWaitReason::kDiagnostic);
      !status.ok()) {
    return status;
  }
  absl::StatusOr<FencePoll> poll = fence->Poll();
  if (!poll.ok()) return poll.status();
  if (poll->state != FenceState::kComplete) {
    return absl::InternalError("cuda.self_test: completion fence did not complete");
  }
  if (absl::Status status = fence->Acknowledge(); !status.ok()) return status;
  if (absl::Status status = compute_done->Release(); !status.ok()) return status;
  if (!std::equal(input_view.HostBytes()->begin(), input_view.HostBytes()->end(),
                  output_view.HostBytes()->begin())) {
    return absl::DataLossError("cuda.self_test: output bytes differ");
  }
  if (absl::Status status = metadata->Release(); !status.ok()) return status;
  if (absl::Status status = workspace->Release(); !status.ok()) return status;
  if (absl::Status status = output->Release(); !status.ok()) return status;
  if (absl::Status status = input->Release(); !status.ok()) return status;
  return context->Shutdown(DeadlineAfter(std::chrono::seconds(5)));
}

absl::Status Stress(const CudaDeviceInfo& info) {
  const auto& api = inferx::cuda::CudaApi::Production();
  const MemoryLimit limits[] = {
      {Device::Host(), MemoryKind::kPinnedHost, ByteCount(kPinnedBudget)},
      {Device::Cuda(info.ordinal), MemoryKind::kDevice, ByteCount(kDeviceBudget)}};
  absl::StatusOr<MemoryTracker> tracker = MemoryTracker::Create(limits);
  if (!tracker.ok()) return tracker.status();
  absl::StatusOr<std::unique_ptr<CudaDeviceContext>> created =
      CudaDeviceContext::Create(TestContextConfig(info.ordinal), *tracker, api);
  if (!created.ok()) return created.status();
  std::unique_ptr<CudaDeviceContext> context = std::move(*created);

  for (uint32_t cycle = 0; cycle < 100000; ++cycle) {
    absl::StatusOr<BufferLease> staging = context->staging_pool().Acquire();
    if (!staging.ok()) return staging.status();
    absl::StatusOr<inferx::cuda::CudaMetadataLease> metadata = context->metadata_ring().Acquire();
    if (!metadata.ok()) return metadata.status();
    absl::StatusOr<WorkspaceLease> workspace = context->workspace_pool().Acquire();
    if (!workspace.ok()) return workspace.status();
    absl::StatusOr<MutableBufferView> staging_view = staging->mutable_view();
    if (!staging_view.ok()) return staging_view.status();
    absl::StatusOr<std::span<std::byte>> staging_bytes = staging_view->HostBytes();
    if (!staging_bytes.ok()) return staging_bytes.status();
    (*staging_bytes)[0] = static_cast<std::byte>((cycle * 17U) & 0xffU);
    MutableBufferView host_metadata = metadata->host_view().value();
    (*host_metadata.HostBytes())[0] = static_cast<std::byte>(cycle & 0xffU);
    absl::StatusOr<BufferView> device_metadata =
        metadata->SealAndUpload(context->transfer_stream(), context->compute_stream());
    if (!device_metadata.ok()) return device_metadata.status();
    absl::StatusOr<MutableBufferView> scratch =
        workspace->Allocate(ByteCount(64), ByteCount(64), inferx::WorkspaceTag::kTest);
    if (!scratch.ok()) return scratch.status();
    absl::StatusOr<CudaEventLease> completed = context->event_pool().Acquire();
    if (!completed.ok()) return completed.status();
    if (absl::Status status = completed->Record(context->compute_stream()); !status.ok()) {
      return status;
    }
    absl::StatusOr<CompletionFence> fence = completed->IntoFence();
    if (!fence.ok()) return fence.status();
    if (absl::Status status =
            fence->WaitUntil(DeadlineAfter(std::chrono::seconds(5)), FenceWaitReason::kTest);
        !status.ok()) {
      return status;
    }
    absl::StatusOr<FencePoll> poll = fence->Poll();
    if (!poll.ok() || poll->state != FenceState::kComplete) {
      return absl::InternalError("cuda.stress: terminal fence missing");
    }
    if (absl::Status status = fence->Acknowledge(); !status.ok()) return status;
    if (absl::Status status = metadata->Release(); !status.ok()) return status;
    if (absl::Status status = workspace->Release(); !status.ok()) return status;
    if (absl::Status status = staging->Release(); !status.ok()) return status;
    if (absl::Status status = context->metadata_ring().ValidateInvariants(); !status.ok()) {
      return status;
    }
    if (absl::Status status = context->workspace_pool().ValidateInvariants(); !status.ok()) {
      return status;
    }
    if (absl::Status status = context->staging_pool().ValidateInvariants(); !status.ok()) {
      return status;
    }
    if (context->event_pool().available_slots() != 8) {
      return absl::InternalError("cuda.stress: event pool did not return baseline");
    }
  }
  return context->Shutdown(DeadlineAfter(std::chrono::seconds(30)));
}

absl::Status RequireCode(const absl::Status& status, absl::StatusCode expected,
                         std::string_view operation) {
  if (!status.ok() && status.code() == expected) return absl::OkStatus();
  return absl::InternalError(std::string("cuda.failure_fixture.") + std::string(operation) +
                             ": unexpected status " + status.ToString());
}

absl::Status InvalidContractTest(const CudaDeviceInfo& info) {
  const auto& api = inferx::cuda::CudaApi::Production();
  CudaHealth health;
  absl::StatusOr<CudaDeviceGuard> guard = CudaDeviceGuard::Create(info.ordinal, api, &health);
  if (!guard.ok()) return guard.status();
  const MemoryLimit limits[] = {
      {Device::Cuda(info.ordinal), MemoryKind::kDevice, ByteCount(kTestBytes)}};
  absl::StatusOr<MemoryTracker> tracker = MemoryTracker::Create(limits);
  if (!tracker.ok()) return tracker.status();
  CudaDeviceAllocator allocator(info.ordinal, *tracker, health, api);
  absl::StatusOr<Buffer> device =
      allocator.Allocate({Device::Cuda(info.ordinal), MemoryKind::kDevice, ByteCount(kTestBytes),
                          ByteCount(64), MemoryCategory::kTest});
  if (!device.ok()) return device.status();
  CpuAllocator cpu;
  absl::StatusOr<Buffer> pageable =
      cpu.Allocate({Device::Host(), MemoryKind::kHost, ByteCount(kTestBytes), ByteCount(64),
                    MemoryCategory::kTest});
  if (!pageable.ok()) return pageable.status();
  absl::StatusOr<CudaStream> stream =
      CudaStream::Create(info.ordinal, CudaStreamRole::kTransfer, 0, api, &health);
  if (!stream.ok()) return stream.status();
  absl::StatusOr<std::unique_ptr<CudaEventPool>> pool =
      CudaEventPool::Create(info.ordinal, 1, api, &health);
  if (!pool.ok()) return pool.status();

  CudaEventLease event = (*pool)->Acquire().value();
  const inferx::FenceToken stale_token = event.token();
  absl::StatusOr<FencePoll> unrecorded = event.Poll();
  if (absl::Status status = RequireCode(unrecorded.status(), absl::StatusCode::kFailedPrecondition,
                                        "unrecorded-event");
      !status.ok()) {
    return status;
  }
  absl::StatusOr<CudaEventLease> exhausted = (*pool)->Acquire();
  if (absl::Status status =
          RequireCode(exhausted.status(), absl::StatusCode::kResourceExhausted, "event-exhaustion");
      !status.ok()) {
    return status;
  }
  MutableBufferView device_view =
      device->MutableView({ByteCount(0), ByteCount(kTestBytes)}).value();
  BufferView pageable_view = pageable->View({ByteCount(0), ByteCount(kTestBytes)}).value();
  if (absl::Status status = RequireCode(
          CopyAsync({pageable_view, device_view, ByteCount(kTestBytes)}, *stream, api, &health),
          absl::StatusCode::kFailedPrecondition, "pageable-copy");
      !status.ok()) {
    return status;
  }
  if (absl::Status status = RequireCode(
          CopyAsync({pageable_view, device_view, ByteCount(kTestBytes + 1)}, *stream, api, &health),
          absl::StatusCode::kOutOfRange, "copy-range");
      !status.ok()) {
    return status;
  }
  if (absl::Status status = event.Release(); !status.ok()) return status;
  if (absl::Status status = RequireCode((*pool)->Poll(stale_token).status(),
                                        absl::StatusCode::kFailedPrecondition, "stale-event");
      !status.ok()) {
    return status;
  }
  if (absl::Status status = (*pool)->Close(); !status.ok()) return status;
  if (absl::Status status = stream->Close(); !status.ok()) return status;
  if (absl::Status status = device->Release(); !status.ok()) return status;
  if (absl::Status status = pageable->Release(); !status.ok()) return status;
  if (absl::Status status = tracker->ValidateBaseline(); !status.ok()) {
    return status;
  }
  return guard->Restore();
}

}  // namespace

int main(int argc, char** argv) {
  bool json = false;
  bool self_test = false;
  bool stress = false;
  bool invalid = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--json")
      json = true;
    else if (argument == "--self-test")
      self_test = true;
    else if (argument == "--stress")
      stress = true;
    else if (argument == "--invalid")
      invalid = true;
    else {
      std::cerr << "unknown argument: " << argument << '\n';
      return 2;
    }
  }
  const auto devices = inferx::cuda::DiscoverCudaDevices();
  if (!devices.ok()) {
    std::cerr << devices.status() << '\n';
    return 1;
  }
  if (json) {
    std::cout << "{\"schema_version\":2,\"accepted_sms\":[89],"
                 "\"toolkit_version\":"
              << CUDART_VERSION
              << ",\"qualification_config\":{"
                 "\"device_budget_bytes\":"
              << kDeviceBudget
              << ",\"device_reserve_bytes\":0,\"event_pool_slots\":8,"
                 "\"metadata_ring_slots\":2,\"metadata_slot_bytes\":"
              << kTestBytes << ",\"pinned_budget_bytes\":" << kPinnedBudget
              << ",\"staging_pool_slots\":2,\"staging_slot_bytes\":" << kTestBytes
              << ",\"workspace_bytes_per_slot\":1048576,"
                 "\"workspace_slots\":1},\"health\":\"discovery-only\","
                 "\"devices\":[";
    for (size_t index = 0; index < devices->size(); ++index) {
      if (index != 0) std::cout << ',';
      const CudaDeviceInfo& device = (*devices)[index];
      std::cout << "{\"ordinal\":" << device.ordinal.value() << ",\"compute_capability\":\""
                << device.compute_major << '.' << device.compute_minor
                << "\",\"total_memory_bytes\":" << device.total_memory.value()
                << ",\"driver_version\":" << device.driver_version
                << ",\"runtime_version\":" << device.runtime_version
                << ",\"unified_addressing\":" << (device.unified_addressing ? "true" : "false")
                << ",\"host_mapping\":" << (device.can_map_host_memory ? "true" : "false")
                << ",\"managed_memory\":" << (device.managed_memory ? "true" : "false")
                << ",\"stream_priorities\":" << (device.stream_priorities ? "true" : "false")
                << ",\"memory_pools\":" << (device.memory_pools ? "true" : "false")
                << ",\"multiprocessors\":" << device.multiprocessor_count
                << ",\"warp_size\":" << device.warp_size
                << ",\"max_threads_per_block\":" << device.max_threads_per_block
                << ",\"copy_engines\":" << device.async_engine_count << ",\"accepted\":"
                << ((device.compute_major * 10 + device.compute_minor) == kAcceptedSm ? "true"
                                                                                      : "false")
                << '}';
    }
    std::cout << "]}\n";
  } else {
    for (const CudaDeviceInfo& device : *devices) {
      std::cout << "CUDA device " << device.ordinal.value() << ": " << device.name << " (sm_"
                << device.compute_major << device.compute_minor << ", "
                << device.total_memory.value() << " bytes)\n";
    }
  }
  if (self_test) {
    const absl::Status status = SelfTest(devices->front());
    if (!status.ok()) {
      std::cerr << status << '\n';
      return 1;
    }
    std::cout << "CUDA asynchronous self-test passed\n";
  }
  if (stress) {
    const absl::Status status = Stress(devices->front());
    if (!status.ok()) {
      std::cerr << status << '\n';
      return 1;
    }
    std::cout << "CUDA 100000-cycle resource stress passed\n";
  }
  if (invalid) {
    const absl::Status status = InvalidContractTest(devices->front());
    if (!status.ok()) {
      std::cerr << status << '\n';
      return 1;
    }
    std::cout << "CUDA failure-contract fixtures passed\n";
  }
  return 0;
}
