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

// In-place fused residual add + RMSNorm (FlashInfer kernel).
cudaError_t LaunchFlashInferFusedAddRmsNorm(const void* input, const void* residual,
                                            const void* weight, uint64_t tokens, uint64_t hidden,
                                            float epsilon, StorageType dtype, cudaStream_t stream);
// Gemma-style (weight + 1) RMSNorm (FlashInfer kernel, weight_bias = 1).
cudaError_t LaunchFlashInferGemmaRmsNorm(const void* input, const void* weight, void* output,
                                         uint64_t tokens, uint64_t hidden, float epsilon,
                                         StorageType dtype, cudaStream_t stream);
// Per-head Q/K RMSNorm (FlashInfer kernel), in place.
cudaError_t LaunchFlashInferQkRmsNorm(void* query, void* key, const void* query_weight,
                                      const void* key_weight, uint64_t tokens, uint64_t query_heads,
                                      uint64_t kv_heads, uint64_t head_dim, float epsilon,
                                      StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::layernorm

#endif  // INFERX_KERNELS_CUDA_LAYERNORM_KERNELS_H_
