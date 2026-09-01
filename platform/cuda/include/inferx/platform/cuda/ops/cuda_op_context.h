#ifndef INFERX_PLATFORM_CUDA_OPS_CUDA_OP_CONTEXT_H_
#define INFERX_PLATFORM_CUDA_OPS_CUDA_OP_CONTEXT_H_

#include "inferx/base/id.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/cuda_stream.h"

namespace inferx::cuda::ops {

struct CudaOpContext {
  DeviceId device;
  CudaStream& stream;
  CudaHealth* health = nullptr;
};

}  // namespace inferx::cuda::ops

#endif  // INFERX_PLATFORM_CUDA_OPS_CUDA_OP_CONTEXT_H_
