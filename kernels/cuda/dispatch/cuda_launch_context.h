// CUDA-side launch context shared by all providers in kernels/cuda.
#ifndef INFERX_KERNELS_CUDA_DISPATCH_CUDA_LAUNCH_CONTEXT_H_
#define INFERX_KERNELS_CUDA_DISPATCH_CUDA_LAUNCH_CONTEXT_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/base/id.h"
#include "inferx/kernels/kernel_dispatch.h"

namespace inferx::kernels::cuda {

struct CudaLaunchContext {
  cudaStream_t stream = nullptr;
  DeviceId device{0};
  uint16_t compute_capability = 0;
  KernelFailureObserver* observer = nullptr;
};

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_DISPATCH_CUDA_LAUNCH_CONTEXT_H_
