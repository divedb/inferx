// Quantization and MoE-routing launch shims (owned kernels; no external
// provider at the pins).
#ifndef INFERX_KERNELS_CUDA_QUANTIZATION_KERNELS_H_
#define INFERX_KERNELS_CUDA_QUANTIZATION_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/storage_type.h"

namespace inferx::kernels::cuda::quant {

// input FP32 [tokens, dim] -> e4m3 bits (uint8) + FP32 scales.
// granularity: 0 = per tensor, 1 = per token, 2 = per token-group.
cudaError_t LaunchFp8Quant(const void* input, void* output, float* scales, uint64_t tokens,
                           uint64_t dim, uint32_t granularity, uint32_t group_size,
                           cudaStream_t stream);

}  // namespace inferx::kernels::cuda::quant

namespace inferx::kernels::cuda::moe {

// Softmax + top-k + optional renorm: logits [tokens, experts] FP32 ->
// weights FP32 [tokens, k], ids int32 [tokens, k].
cudaError_t LaunchSoftmaxTopK(const void* logits, void* weights, int32_t* ids, uint64_t tokens,
                              uint64_t experts, uint32_t top_k, bool renormalize,
                              cudaStream_t stream);

// Sigmoid routing with additive selection bias (Kimi style).
cudaError_t LaunchSigmoidBiasTopK(const void* logits, const void* bias, void* weights, int32_t* ids,
                                  uint64_t tokens, uint64_t experts, uint32_t top_k,
                                  float routed_scaling_factor, bool renormalize,
                                  cudaStream_t stream);

}  // namespace inferx::kernels::cuda::moe

#endif  // INFERX_KERNELS_CUDA_QUANTIZATION_KERNELS_H_
