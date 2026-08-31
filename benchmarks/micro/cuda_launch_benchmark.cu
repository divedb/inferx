#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

namespace {

__global__ void NoOpKernel() {}

__attribute__((noinline)) cudaError_t LaunchNoOpWrapper(cudaStream_t stream) {
  NoOpKernel<<<1, 1, 0, stream>>>();
  return cudaPeekAtLastError();
}

void RunCudaNoOpLaunch(benchmark::State& state, bool wrapper) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    state.SkipWithError("no CUDA device");
    return;
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    state.SkipWithError("stream creation failed");
    return;
  }
  for (auto _ : state) {
    static_cast<void>(_);
    if (wrapper) {
      if (LaunchNoOpWrapper(stream) != cudaSuccess) {
        state.SkipWithError("wrapper launch rejected");
        break;
      }
    } else {
      NoOpKernel<<<1, 1, 0, stream>>>();
      if (cudaPeekAtLastError() != cudaSuccess) {
        state.SkipWithError("direct launch rejected");
        break;
      }
    }
    benchmark::DoNotOptimize(stream);
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}

void BM_CudaNoOpLaunchDirect(benchmark::State& state) { RunCudaNoOpLaunch(state, false); }

void BM_CudaNoOpLaunchWrapper(benchmark::State& state) { RunCudaNoOpLaunch(state, true); }

BENCHMARK(BM_CudaNoOpLaunchDirect)->Repetitions(30);
BENCHMARK(BM_CudaNoOpLaunchWrapper)->Repetitions(30);

}  // namespace

BENCHMARK_MAIN();
