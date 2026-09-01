#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"
#include "inferx/platform/cuda/cuda_allocator.h"
#include "inferx/platform/cuda/cuda_buffer_access.h"
#include "inferx/platform/cuda/cuda_copy.h"
#include "inferx/platform/cuda/cuda_device.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/runtime/memory_tracker.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"

namespace {

enum class CopyCase : uint8_t {
  kPinnedH2DWrapper,
  kPinnedH2DDirect,
  kPageableH2DRawBenchmarkAdapter,
  kPinnedD2HWrapper,
  kPinnedD2HDirect,
  kD2DWrapper,
  kD2DDirect,
};

bool RangesOverlap(inferx::ByteRange left, inferx::ByteRange right) {
  if (left.size.value() == 0 || right.size.value() == 0) return false;
  return left.offset.value() < right.offset.value() + right.size.value() &&
         right.offset.value() < left.offset.value() + left.size.value();
}

// Mirrors CopyAsync's release-mode safety contract around a direct cudart
// submission. Keeping this out of line prevents constant fixture metadata
// from being folded away only in the direct baseline.
__attribute__((noinline)) absl::Status DirectCopyAsyncEquivalent(
    const inferx::cuda::CopyRequest& request, inferx::cuda::CudaStream& stream,
    inferx::cuda::CudaHealth& health) {
  absl::Status accepting = health.CheckAcceptingWork();
  if (!accepting.ok()) return accepting;
  if (request.bytes.value() > request.source.size().value() ||
      request.bytes.value() > request.destination.size().value()) {
    return absl::OutOfRangeError("benchmark.copy.bytes: copy exceeds a view");
  }
  if (request.bytes.value() == 0) return absl::OkStatus();
  const inferx::MemoryKind source_kind = request.source.memory_kind();
  const inferx::MemoryKind destination_kind = request.destination.memory_kind();
  if (source_kind == inferx::MemoryKind::kManaged ||
      destination_kind == inferx::MemoryKind::kManaged) {
    return absl::UnimplementedError("benchmark.copy: managed memory is unsupported");
  }
  if (source_kind == inferx::MemoryKind::kHost || destination_kind == inferx::MemoryKind::kHost) {
    return absl::FailedPreconditionError("benchmark.copy: pageable memory is unsupported");
  }
  cudaMemcpyKind direction = cudaMemcpyDefault;
  if (source_kind == inferx::MemoryKind::kPinnedHost &&
      destination_kind == inferx::MemoryKind::kDevice) {
    direction = cudaMemcpyHostToDevice;
    if (request.destination.device() != inferx::Device::Cuda(stream.device())) {
      return absl::InvalidArgumentError("benchmark.copy: destination device mismatch");
    }
  } else if (source_kind == inferx::MemoryKind::kDevice &&
             destination_kind == inferx::MemoryKind::kPinnedHost) {
    direction = cudaMemcpyDeviceToHost;
    if (request.source.device() != inferx::Device::Cuda(stream.device())) {
      return absl::InvalidArgumentError("benchmark.copy: source device mismatch");
    }
  } else if (source_kind == inferx::MemoryKind::kDevice &&
             destination_kind == inferx::MemoryKind::kDevice) {
    direction = cudaMemcpyDeviceToDevice;
    if (request.source.device() != request.destination.device() ||
        request.source.device() != inferx::Device::Cuda(stream.device())) {
      return absl::InvalidArgumentError("benchmark.copy: D2D device mismatch");
    }
  } else {
    return absl::InvalidArgumentError("benchmark.copy: unsupported direction");
  }
  if (request.source.allocation_id() == request.destination.allocation_id() &&
      RangesOverlap({request.source.range().offset, request.bytes},
                    {request.destination.range().offset, request.bytes})) {
    return absl::InvalidArgumentError("benchmark.copy: overlapping copy");
  }
  for (const inferx::BufferView view : {request.source, request.destination.AsConst()}) {
    cudaPointerAttributes attributes{};
    const cudaError_t error =
        cudaPointerGetAttributes(&attributes, inferx::cuda::BufferAccess::Address(view));
    if (error != cudaSuccess) {
      return inferx::cuda::CudaErrorStatus(error, "benchmark-pointer-attributes", stream.device(),
                                           &health);
    }
    const cudaMemoryType expected = view.memory_kind() == inferx::MemoryKind::kDevice
                                        ? cudaMemoryTypeDevice
                                        : cudaMemoryTypeHost;
    if (attributes.type != expected) {
      return absl::InvalidArgumentError("benchmark.copy: runtime pointer kind mismatch");
    }
  }
  absl::StatusOr<size_t> bytes =
      inferx::CheckedNarrow<size_t>(request.bytes.value(), "benchmark.copy.bytes");
  if (!bytes.ok()) return bytes.status();
  return inferx::cuda::CudaErrorStatus(
      cudaMemcpyAsync(inferx::cuda::BufferAccess::Address(request.destination),
                      inferx::cuda::BufferAccess::Address(request.source), *bytes, direction,
                      stream.handle()),
      "benchmark-memcpy-async", stream.device(), &health);
}

void RunCudaCopy(benchmark::State& state, CopyCase copy_case) {
  const auto devices = inferx::cuda::DiscoverCudaDevices();
  if (!devices.ok() || devices->empty()) {
    state.SkipWithError("no CUDA device");
    return;
  }
  const inferx::DeviceId device_id = devices->front().ordinal;
  auto guard = inferx::cuda::CudaDeviceGuard::Create(device_id);
  if (!guard.ok()) {
    state.SkipWithError("device guard failed");
    return;
  }
  const uint64_t bytes = static_cast<uint64_t>(state.range(0));
  const std::array<inferx::MemoryLimit, 2> limits{
      inferx::MemoryLimit{inferx::Device::Host(), inferx::MemoryKind::kPinnedHost,
                          inferx::ByteCount(bytes * 2)},
      inferx::MemoryLimit{inferx::Device::Cuda(device_id), inferx::MemoryKind::kDevice,
                          inferx::ByteCount(bytes * 2)}};
  auto tracker_result = inferx::MemoryTracker::Create(limits);
  if (!tracker_result.ok()) {
    state.SkipWithError("tracker setup failed");
    return;
  }
  inferx::MemoryTracker tracker = std::move(*tracker_result);
  inferx::cuda::CudaHealth health;
  inferx::cuda::CudaPinnedAllocator pinned(device_id, tracker, health);
  inferx::cuda::CudaDeviceAllocator device(device_id, tracker, health);
  inferx::CpuAllocator cpu;
  inferx::Buffer first;
  inferx::Buffer second;

  const inferx::AllocationRequest pinned_request{
      inferx::Device::Host(), inferx::MemoryKind::kPinnedHost, inferx::ByteCount(bytes),
      inferx::ByteCount(64), inferx::MemoryCategory::kTest};
  const inferx::AllocationRequest device_request{
      inferx::Device::Cuda(device_id), inferx::MemoryKind::kDevice, inferx::ByteCount(bytes),
      inferx::ByteCount(64), inferx::MemoryCategory::kTest};
  const inferx::AllocationRequest pageable_request{
      inferx::Device::Host(), inferx::MemoryKind::kHost, inferx::ByteCount(bytes),
      inferx::ByteCount(64), inferx::MemoryCategory::kTest};
  absl::StatusOr<inferx::Buffer> first_result =
      copy_case == CopyCase::kPageableH2DRawBenchmarkAdapter
          ? cpu.Allocate(pageable_request)
          : (copy_case == CopyCase::kD2DWrapper || copy_case == CopyCase::kD2DDirect
                 ? device.Allocate(device_request)
                 : (copy_case == CopyCase::kPinnedD2HWrapper ||
                            copy_case == CopyCase::kPinnedD2HDirect
                        ? device.Allocate(device_request)
                        : pinned.Allocate(pinned_request)));
  absl::StatusOr<inferx::Buffer> second_result =
      copy_case == CopyCase::kPinnedD2HWrapper || copy_case == CopyCase::kPinnedD2HDirect
          ? pinned.Allocate(pinned_request)
          : device.Allocate(device_request);
  if (!first_result.ok() || !second_result.ok()) {
    state.SkipWithError("CUDA copy allocation setup failed");
    return;
  }
  first = std::move(*first_result);
  second = std::move(*second_result);
  const inferx::BufferView source =
      first.View({inferx::ByteCount(0), inferx::ByteCount(bytes)}).value();
  const inferx::MutableBufferView destination =
      second.MutableView({inferx::ByteCount(0), inferx::ByteCount(bytes)}).value();
  auto stream = inferx::cuda::CudaStream::Create(device_id, inferx::cuda::CudaStreamRole::kTransfer,
                                                 0, inferx::cuda::CudaApi::Production(), &health);
  if (!stream.ok()) {
    state.SkipWithError("CUDA stream setup failed");
    return;
  }
  cudaEvent_t timing_start = nullptr;
  cudaEvent_t timing_stop = nullptr;
  if (cudaEventCreate(&timing_start) != cudaSuccess ||
      cudaEventCreate(&timing_stop) != cudaSuccess) {
    if (timing_start != nullptr) cudaEventDestroy(timing_start);
    state.SkipWithError("CUDA timing event setup failed");
    stream->Close().IgnoreError();
    second.Release().IgnoreError();
    first.Release().IgnoreError();
    guard->Restore().IgnoreError();
    return;
  }

  const bool wrapper = copy_case == CopyCase::kPinnedH2DWrapper ||
                       copy_case == CopyCase::kPinnedD2HWrapper ||
                       copy_case == CopyCase::kD2DWrapper;

  auto submit_copy = [&]() -> bool {
    if (wrapper) {
      return inferx::cuda::CopyAsync({source, destination, inferx::ByteCount(bytes)}, *stream,
                                     inferx::cuda::CudaApi::Production(), &health)
          .ok();
    }
    if (copy_case == CopyCase::kPageableH2DRawBenchmarkAdapter) {
      return cudaMemcpyAsync(inferx::cuda::BufferAccess::Address(destination),
                             static_cast<const void*>(source.HostBytes()->data()),
                             static_cast<size_t>(bytes), cudaMemcpyHostToDevice,
                             stream->handle()) == cudaSuccess;
    }
    return DirectCopyAsyncEquivalent({source, destination, inferx::ByteCount(bytes)}, *stream,
                                     health)
        .ok();
  };

  // Submission time intentionally excludes device completion. Use an explicit
  // warm-up and a bounded iteration count so Google Benchmark does not
  // calibrate hundreds of thousands of synchronized 256 MiB transfers from
  // the much smaller host-submission duration.
  constexpr int kWarmupCopies = 8;
  bool warmup_ok = true;
  for (int index = 0; index < kWarmupCopies; ++index) {
    warmup_ok = warmup_ok && submit_copy();
  }
  warmup_ok = warmup_ok && cudaStreamSynchronize(stream->handle()) == cudaSuccess;
  if (!warmup_ok) state.SkipWithError("copy warm-up failed");

  double device_milliseconds = 0.0;
  int64_t completed_iterations = 0;
  if (warmup_ok) {
    for (auto _ : state) {
      static_cast<void>(_);
      state.PauseTiming();
      if (cudaEventRecord(timing_start, stream->handle()) != cudaSuccess) {
        state.ResumeTiming();
        state.SkipWithError("copy timing start failed");
        break;
      }
      state.ResumeTiming();
      if (!submit_copy()) {
        state.SkipWithError("copy submission failed");
        break;
      }
      state.PauseTiming();
      float elapsed_milliseconds = 0.0F;
      const cudaError_t stop_error = cudaEventRecord(timing_stop, stream->handle());
      const cudaError_t completion_error =
          stop_error == cudaSuccess ? cudaEventSynchronize(timing_stop) : stop_error;
      const cudaError_t elapsed_error =
          completion_error == cudaSuccess
              ? cudaEventElapsedTime(&elapsed_milliseconds, timing_start, timing_stop)
              : completion_error;
      state.ResumeTiming();
      if (elapsed_error != cudaSuccess) {
        state.SkipWithError("copy device timing failed");
        break;
      }
      device_milliseconds += static_cast<double>(elapsed_milliseconds);
      ++completed_iterations;
    }
  }
  state.SetBytesProcessed(completed_iterations * static_cast<int64_t>(bytes));
  state.counters["cuda_memcpy_async_calls"] = static_cast<double>(completed_iterations);
  state.counters["cuda_pointer_attribute_calls"] =
      copy_case == CopyCase::kPageableH2DRawBenchmarkAdapter
          ? 0.0
          : static_cast<double>(completed_iterations * 2);
  state.counters["cuda_timing_event_sync_calls"] = static_cast<double>(completed_iterations);
  if (device_milliseconds > 0.0) {
    constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    state.counters["device_GiB_per_second"] =
        (static_cast<double>(completed_iterations) * static_cast<double>(bytes) / kBytesPerGiB) /
        (device_milliseconds / 1000.0);
  }
  cudaEventDestroy(timing_stop);
  cudaEventDestroy(timing_start);
  stream->Close().IgnoreError();
  second.Release().IgnoreError();
  first.Release().IgnoreError();
  if (!tracker.ValidateBaseline().ok()) {
    state.SkipWithError("copy accounting did not return to baseline");
  }
  guard->Restore().IgnoreError();
}

void BM_CudaPinnedH2DWrapper(benchmark::State& state) {
  RunCudaCopy(state, CopyCase::kPinnedH2DWrapper);
}
void BM_CudaPinnedH2DDirect(benchmark::State& state) {
  RunCudaCopy(state, CopyCase::kPinnedH2DDirect);
}
void BM_CudaPageableH2DRawBenchmarkAdapter(benchmark::State& state) {
  RunCudaCopy(state, CopyCase::kPageableH2DRawBenchmarkAdapter);
}
void BM_CudaPinnedD2HWrapper(benchmark::State& state) {
  RunCudaCopy(state, CopyCase::kPinnedD2HWrapper);
}
void BM_CudaPinnedD2HDirect(benchmark::State& state) {
  RunCudaCopy(state, CopyCase::kPinnedD2HDirect);
}
void BM_CudaD2DWrapper(benchmark::State& state) { RunCudaCopy(state, CopyCase::kD2DWrapper); }
void BM_CudaD2DDirect(benchmark::State& state) { RunCudaCopy(state, CopyCase::kD2DDirect); }

using CopyBenchmark = void (*)(benchmark::State&);

void RegisterCopyCase(const char* name, CopyBenchmark function, int64_t bytes, int64_t iterations) {
  benchmark::RegisterBenchmark(name, function)
      ->Arg(bytes)
      ->Iterations(iterations)
      ->Repetitions(30)
      ->UseRealTime();
}

[[maybe_unused]] const bool kCopyBenchmarksRegistered = [] {
  constexpr std::array<int64_t, 7> kSizes{4096,     32768,     262144,   2097152,
                                          16777216, 134217728, 268435456};
  for (const int64_t bytes : kSizes) {
    const int64_t iterations = bytes == 4096 ? 4096 : 32;
    RegisterCopyCase("BM_CudaPinnedH2DWrapper", BM_CudaPinnedH2DWrapper, bytes, iterations);
    RegisterCopyCase("BM_CudaPinnedH2DDirect", BM_CudaPinnedH2DDirect, bytes, iterations);
    RegisterCopyCase("BM_CudaPageableH2DRawBenchmarkAdapter", BM_CudaPageableH2DRawBenchmarkAdapter,
                     bytes, iterations);
    RegisterCopyCase("BM_CudaPinnedD2HWrapper", BM_CudaPinnedD2HWrapper, bytes, iterations);
    RegisterCopyCase("BM_CudaPinnedD2HDirect", BM_CudaPinnedD2HDirect, bytes, iterations);
    RegisterCopyCase("BM_CudaD2DWrapper", BM_CudaD2DWrapper, bytes, iterations);
    RegisterCopyCase("BM_CudaD2DDirect", BM_CudaD2DDirect, bytes, iterations);
  }
  return true;
}();

}  // namespace

BENCHMARK_MAIN();
