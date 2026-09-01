#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

__global__ void NoOpKernel() {}

__global__ void ByteCopyKernel(std::byte* destination, const std::byte* source, size_t bytes) {
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < bytes;
       index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    destination[index] = source[index];
  }
}

__attribute__((noinline)) cudaError_t LaunchNoOpWrapper(cudaStream_t stream) {
  NoOpKernel<<<1, 1, 0, stream>>>();
  return cudaPeekAtLastError();
}

__attribute__((noinline)) cudaError_t LaunchByteCopyWrapper(std::byte* destination,
                                                            const std::byte* source, size_t bytes,
                                                            cudaStream_t stream) {
  constexpr unsigned int kThreads = 256;
  const unsigned int blocks = static_cast<unsigned int>((bytes + kThreads - 1) / kThreads);
  ByteCopyKernel<<<blocks, kThreads, 0, stream>>>(destination, source, bytes);
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
  int64_t completed_iterations = 0;
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
    ++completed_iterations;
  }
  state.SetItemsProcessed(completed_iterations);
  state.counters["cuda_kernel_launch_calls"] = static_cast<double>(completed_iterations);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}

void BM_CudaNoOpLaunchDirect(benchmark::State& state) { RunCudaNoOpLaunch(state, false); }

void BM_CudaNoOpLaunchWrapper(benchmark::State& state) { RunCudaNoOpLaunch(state, true); }

void RunCudaByteCopyLaunch(benchmark::State& state, bool wrapper) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    state.SkipWithError("no CUDA device");
    return;
  }
  const size_t bytes = static_cast<size_t>(state.range(0));
  std::byte* source = nullptr;
  std::byte* destination = nullptr;
  cudaStream_t stream = nullptr;
  if (cudaMalloc(&source, bytes) != cudaSuccess || cudaMalloc(&destination, bytes) != cudaSuccess ||
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    if (destination != nullptr) cudaFree(destination);
    if (source != nullptr) cudaFree(source);
    state.SkipWithError("byte-copy launch setup failed");
    return;
  }
  constexpr unsigned int kThreads = 256;
  const unsigned int blocks = static_cast<unsigned int>((bytes + kThreads - 1) / kThreads);
  int64_t completed_iterations = 0;
  for (auto _ : state) {
    static_cast<void>(_);
    if (wrapper) {
      if (LaunchByteCopyWrapper(destination, source, bytes, stream) != cudaSuccess) {
        state.SkipWithError("wrapper byte-copy launch rejected");
        break;
      }
    } else {
      ByteCopyKernel<<<blocks, kThreads, 0, stream>>>(destination, source, bytes);
      if (cudaPeekAtLastError() != cudaSuccess) {
        state.SkipWithError("direct byte-copy launch rejected");
        break;
      }
    }
    benchmark::DoNotOptimize(stream);
    ++completed_iterations;
  }
  state.SetItemsProcessed(completed_iterations);
  state.counters["cuda_kernel_launch_calls"] = static_cast<double>(completed_iterations);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  cudaFree(destination);
  cudaFree(source);
}

void BM_CudaByteCopyLaunchDirect(benchmark::State& state) { RunCudaByteCopyLaunch(state, false); }

void BM_CudaByteCopyLaunchWrapper(benchmark::State& state) { RunCudaByteCopyLaunch(state, true); }

BENCHMARK(BM_CudaNoOpLaunchDirect)->Repetitions(30)->MinWarmUpTime(0.1)->UseRealTime();
BENCHMARK(BM_CudaNoOpLaunchWrapper)->Repetitions(30)->MinWarmUpTime(0.1)->UseRealTime();
BENCHMARK(BM_CudaByteCopyLaunchDirect)
    ->Arg(4096)
    ->Arg(1024 * 1024)
    ->Repetitions(30)
    ->MinWarmUpTime(0.1)
    ->UseRealTime();
BENCHMARK(BM_CudaByteCopyLaunchWrapper)
    ->Arg(4096)
    ->Arg(1024 * 1024)
    ->Repetitions(30)
    ->MinWarmUpTime(0.1)
    ->UseRealTime();

}  // namespace

BENCHMARK_MAIN();
