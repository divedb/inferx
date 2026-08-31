#include <benchmark/benchmark.h>

#include <memory>

#include "inferx/base/clock.h"
#include "inferx/platform/cuda/cuda_device.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/cuda_event_pool.h"
#include "inferx/platform/cuda/cuda_stream.h"

namespace {

void BM_CudaEventRecordWaitAcknowledge(benchmark::State& state) {
  auto devices = inferx::cuda::DiscoverCudaDevices();
  if (!devices.ok() || devices->empty()) {
    state.SkipWithError("no CUDA device");
    return;
  }
  const inferx::DeviceId device = devices->front().ordinal;
  inferx::cuda::CudaHealth health;
  auto guard = inferx::cuda::CudaDeviceGuard::Create(device);
  auto stream = inferx::cuda::CudaStream::Create(device, inferx::cuda::CudaStreamRole::kCompute, 0,
                                                 inferx::cuda::CudaApi::Production(), &health);
  auto pool =
      inferx::cuda::CudaEventPool::Create(device, 8, inferx::cuda::CudaApi::Production(), &health);
  if (!guard.ok() || !stream.ok() || !pool.ok()) {
    state.SkipWithError("CUDA setup failed");
    return;
  }
  for (auto _ : state) {
    static_cast<void>(_);
    auto event = (*pool)->Acquire().value();
    event.Record(*stream).IgnoreError();
    auto fence = event.IntoFence().value();
    while (fence.Poll()->state == inferx::FenceState::kPending) {
    }
    fence.Acknowledge().IgnoreError();
  }
  (*pool)->Close().IgnoreError();
  stream->Close().IgnoreError();
  guard->Restore().IgnoreError();
}

void BM_CudaEventRecordQueryDirect(benchmark::State& state) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    state.SkipWithError("no CUDA device");
    return;
  }
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
    state.SkipWithError("direct CUDA event setup failed");
    if (stream != nullptr) cudaStreamDestroy(stream);
    return;
  }
  for (auto _ : state) {
    static_cast<void>(_);
    if (cudaEventRecord(event, stream) != cudaSuccess) {
      state.SkipWithError("direct event record failed");
      break;
    }
    cudaError_t query = cudaErrorNotReady;
    while (query == cudaErrorNotReady) query = cudaEventQuery(event);
    if (query != cudaSuccess) {
      state.SkipWithError("direct event query failed");
      break;
    }
  }
  cudaEventDestroy(event);
  cudaStreamDestroy(stream);
}

BENCHMARK(BM_CudaEventRecordWaitAcknowledge)->Repetitions(30);
BENCHMARK(BM_CudaEventRecordQueryDirect)->Repetitions(30);

}  // namespace

BENCHMARK_MAIN();
