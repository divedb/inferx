#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "inferx/platform/cuda/cuda_allocator.h"
#include "inferx/platform/cuda/cuda_buffer_access.h"
#include "inferx/platform/cuda/cuda_copy.h"
#include "inferx/platform/cuda/cuda_device.h"
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

  const bool wrapper = copy_case == CopyCase::kPinnedH2DWrapper ||
                       copy_case == CopyCase::kPinnedD2HWrapper ||
                       copy_case == CopyCase::kD2DWrapper;
  cudaMemcpyKind direction = cudaMemcpyHostToDevice;
  if (copy_case == CopyCase::kPinnedD2HWrapper || copy_case == CopyCase::kPinnedD2HDirect) {
    direction = cudaMemcpyDeviceToHost;
  } else if (copy_case == CopyCase::kD2DWrapper || copy_case == CopyCase::kD2DDirect) {
    direction = cudaMemcpyDeviceToDevice;
  }

  for (auto _ : state) {
    static_cast<void>(_);
    if (wrapper) {
      const absl::Status status =
          inferx::cuda::CopyAsync({source, destination, inferx::ByteCount(bytes)}, *stream,
                                  inferx::cuda::CudaApi::Production(), &health);
      if (!status.ok()) {
        state.SkipWithError("wrapper copy submission failed");
        break;
      }
    } else {
      const void* source_address = copy_case == CopyCase::kPageableH2DRawBenchmarkAdapter
                                       ? static_cast<const void*>(source.HostBytes()->data())
                                       : inferx::cuda::BufferAccess::Address(source);
      const cudaError_t error =
          cudaMemcpyAsync(inferx::cuda::BufferAccess::Address(destination), source_address,
                          static_cast<size_t>(bytes), direction, stream->handle());
      if (error != cudaSuccess) {
        state.SkipWithError("direct copy submission failed");
        break;
      }
    }
    if (cudaStreamSynchronize(stream->handle()) != cudaSuccess) {
      state.SkipWithError("copy completion failed");
      break;
    }
  }
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes));
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

#define INFERX_REGISTER_COPY_BENCHMARK(function_name) \
  BENCHMARK(function_name)->RangeMultiplier(8)->Range(4096, 256 * 1024 * 1024)->Repetitions(30)

INFERX_REGISTER_COPY_BENCHMARK(BM_CudaPinnedH2DWrapper);
INFERX_REGISTER_COPY_BENCHMARK(BM_CudaPinnedH2DDirect);
INFERX_REGISTER_COPY_BENCHMARK(BM_CudaPageableH2DRawBenchmarkAdapter);
INFERX_REGISTER_COPY_BENCHMARK(BM_CudaPinnedD2HWrapper);
INFERX_REGISTER_COPY_BENCHMARK(BM_CudaPinnedD2HDirect);
INFERX_REGISTER_COPY_BENCHMARK(BM_CudaD2DWrapper);
INFERX_REGISTER_COPY_BENCHMARK(BM_CudaD2DDirect);

#undef INFERX_REGISTER_COPY_BENCHMARK

}  // namespace

BENCHMARK_MAIN();
