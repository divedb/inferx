#include <benchmark/benchmark.h>

#include <cstdint>
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
  if (!guard.ok()) {
    state.SkipWithError("CUDA device guard failed");
    return;
  }
  auto producer = inferx::cuda::CudaStream::Create(device, inferx::cuda::CudaStreamRole::kCompute,
                                                   0, inferx::cuda::CudaApi::Production(), &health);
  auto consumer = inferx::cuda::CudaStream::Create(device, inferx::cuda::CudaStreamRole::kTransfer,
                                                   0, inferx::cuda::CudaApi::Production(), &health);
  auto pool =
      inferx::cuda::CudaEventPool::Create(device, 8, inferx::cuda::CudaApi::Production(), &health);
  if (!producer.ok() || !consumer.ok() || !pool.ok()) {
    state.SkipWithError("CUDA setup failed");
    if (pool.ok()) (*pool)->Close().IgnoreError();
    if (consumer.ok()) consumer->Close().IgnoreError();
    if (producer.ok()) producer->Close().IgnoreError();
    guard->Restore().IgnoreError();
    return;
  }
  int64_t completed_iterations = 0;
  int64_t query_calls = 0;
  for (auto _ : state) {
    static_cast<void>(_);
    auto event = (*pool)->Acquire().value();
    if (!event.Record(*producer).ok() || !event.WaitOn(*consumer).ok()) {
      state.SkipWithError("wrapper event submission failed");
      break;
    }
    auto fence = event.IntoFence().value();
    bool poll_failed = false;
    while (true) {
      auto poll = fence.Poll();
      ++query_calls;
      if (!poll.ok()) {
        state.SkipWithError("wrapper event query failed");
        poll_failed = true;
        break;
      }
      if (poll->state != inferx::FenceState::kPending) break;
    }
    if (poll_failed) break;
    if (!fence.Acknowledge().ok()) {
      state.SkipWithError("wrapper event acknowledgement failed");
      break;
    }
    ++completed_iterations;
  }
  state.SetItemsProcessed(completed_iterations);
  state.counters["cuda_event_record_calls"] = static_cast<double>(completed_iterations);
  state.counters["cuda_event_query_calls"] = static_cast<double>(query_calls);
  state.counters["cuda_stream_wait_event_calls"] = static_cast<double>(completed_iterations);
  (*pool)->Close().IgnoreError();
  consumer->Close().IgnoreError();
  producer->Close().IgnoreError();
  guard->Restore().IgnoreError();
}

void BM_CudaEventRecordWaitQueryDirect(benchmark::State& state) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    state.SkipWithError("no CUDA device");
    return;
  }
  cudaStream_t producer = nullptr;
  cudaStream_t consumer = nullptr;
  cudaEvent_t event = nullptr;
  if (cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) != cudaSuccess ||
      cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
    state.SkipWithError("direct CUDA event setup failed");
    if (consumer != nullptr) cudaStreamDestroy(consumer);
    if (producer != nullptr) cudaStreamDestroy(producer);
    return;
  }
  int64_t completed_iterations = 0;
  int64_t query_calls = 0;
  for (auto _ : state) {
    static_cast<void>(_);
    if (cudaEventRecord(event, producer) != cudaSuccess ||
        cudaStreamWaitEvent(consumer, event, 0) != cudaSuccess) {
      state.SkipWithError("direct event record failed");
      break;
    }
    cudaError_t query = cudaErrorNotReady;
    while (query == cudaErrorNotReady) {
      query = cudaEventQuery(event);
      ++query_calls;
    }
    if (query != cudaSuccess) {
      state.SkipWithError("direct event query failed");
      break;
    }
    ++completed_iterations;
  }
  state.SetItemsProcessed(completed_iterations);
  state.counters["cuda_event_record_calls"] = static_cast<double>(completed_iterations);
  state.counters["cuda_event_query_calls"] = static_cast<double>(query_calls);
  state.counters["cuda_stream_wait_event_calls"] = static_cast<double>(completed_iterations);
  cudaEventDestroy(event);
  cudaStreamDestroy(consumer);
  cudaStreamDestroy(producer);
}

BENCHMARK(BM_CudaEventRecordWaitAcknowledge)->Repetitions(30)->MinWarmUpTime(0.1)->UseRealTime();
BENCHMARK(BM_CudaEventRecordWaitQueryDirect)->Repetitions(30)->MinWarmUpTime(0.1)->UseRealTime();

}  // namespace

BENCHMARK_MAIN();
