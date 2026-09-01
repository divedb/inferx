#include "inferx/platform/cuda/cuda_api.h"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace inferx::cuda {

const CudaApi& CudaApi::Production() noexcept {
  static const CudaApi api{
      &cudaGetErrorName,
      &cudaGetErrorString,
      &cudaGetDeviceCount,
      &cudaGetDevice,
      &cudaSetDevice,
      &cudaGetDeviceProperties,
      &cudaDriverGetVersion,
      &cudaRuntimeGetVersion,
      &cudaMemGetInfo,
      &cudaDeviceGetStreamPriorityRange,
      &cudaStreamCreateWithPriority,
      &cudaStreamDestroy,
      &cudaStreamWaitEvent,
      &cudaEventCreateWithFlags,
      &cudaEventDestroy,
      &cudaEventRecord,
      &cudaEventQuery,
      static_cast<cudaError_t (*)(void**, size_t)>(&cudaMalloc),
      &cudaFree,
      &cudaHostAlloc,
      &cudaFreeHost,
      &cudaMemcpyAsync,
      &cudaMemsetAsync,
      &cudaPointerGetAttributes,
      &cudaPeekAtLastError,
      &cudaDeviceSynchronize,
  };
  return api;
}

}  // namespace inferx::cuda
