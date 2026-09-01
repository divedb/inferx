// Layernorm category launch shim (FlashInfer RMSNorm instantiation).
#ifndef INFERX_KERNELS_CUDA_LAYERNORM_KERNELS_H_
#define INFERX_KERNELS_CUDA_LAYERNORM_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/storage_type.h"

namespace inferx::kernels::cuda::layernorm {

// FlashInfer's RMSNormKernel requires hidden % vec_size == 0 (vec is 8 for
// FP16/BF16, 4 for FP32); other shapes are rejected as invalid here and
// surface as Unimplemented from the provider layer.
cudaError_t LaunchFlashInferRmsNorm(const void* input, const void* weight, void* output,
                                    uint64_t tokens, uint64_t hidden, float epsilon,
                                    StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::layernorm

#endif  // INFERX_KERNELS_CUDA_LAYERNORM_KERNELS_H_
