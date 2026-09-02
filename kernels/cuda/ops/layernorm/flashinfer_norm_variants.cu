// FlashInfer layernorm variants (ADR 0032 reuse tier 2): fused
// residual-add RMSNorm, Gemma-style (weight+1) RMSNorm, and per-head Q/K
// RMSNorm — all ahead-of-time kernel instantiations.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "flashinfer/norm.cuh"
#include "inferx/kernels/cuda/layernorm_kernels.h"

namespace inferx::kernels::cuda::layernorm {
namespace {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kMaxWarps = 32;

uint32_t WarpsFor(uint64_t width, uint32_t vec_size) {
  const uint64_t lanes = (width + vec_size - 1) / vec_size;
  uint64_t warps = (lanes + kWarpSize - 1) / kWarpSize;
  if (warps == 0) warps = 1;
  if (warps > kMaxWarps) warps = kMaxWarps;
  return static_cast<uint32_t>(warps);
}

template <typename T, uint32_t kVecSize>
cudaError_t LaunchFusedAddTyped(const void* input, const void* residual, const void* weight,
                                uint64_t tokens, uint64_t hidden, float epsilon,
                                cudaStream_t stream) {
  if (hidden % kVecSize != 0) return cudaErrorInvalidValue;
  const uint32_t num_warps = WarpsFor(hidden, kVecSize);
  // The kernel stages the combined row in shared memory after the
  // reduction slots (flashinfer host launcher: (ceil4(warps) + d) floats).
  const size_t shared_bytes =
      (((num_warps + 3) / 4) * 4 + static_cast<size_t>(hidden)) * sizeof(float);
  flashinfer::norm::FusedAddRMSNormKernel<kVecSize, T>
      <<<dim3(static_cast<uint32_t>(tokens)), dim3(kWarpSize, num_warps), shared_bytes, stream>>>(
          static_cast<T*>(const_cast<void*>(input)),
          static_cast<T*>(const_cast<void*>(residual)),
          static_cast<T*>(const_cast<void*>(weight)), static_cast<uint32_t>(hidden),
          static_cast<uint32_t>(hidden), static_cast<uint32_t>(hidden),
          /*weight_bias=*/0.0F, epsilon);
  return cudaGetLastError();
}

template <typename T, uint32_t kVecSize>
cudaError_t LaunchGemmaTyped(const void* input, const void* weight, void* output,
                             uint64_t tokens, uint64_t hidden, float epsilon,
                             cudaStream_t stream) {
  if (hidden % kVecSize != 0) return cudaErrorInvalidValue;
  const uint32_t num_warps = WarpsFor(hidden, kVecSize);
  const size_t shared_bytes = num_warps * sizeof(float);
  flashinfer::norm::RMSNormKernel<kVecSize, T>
      <<<dim3(static_cast<uint32_t>(tokens)), dim3(kWarpSize, num_warps), shared_bytes, stream>>>(
          static_cast<T*>(const_cast<void*>(input)),
          static_cast<T*>(const_cast<void*>(weight)), static_cast<T*>(output),
          static_cast<uint32_t>(hidden), static_cast<uint32_t>(hidden),
          static_cast<uint32_t>(hidden), /*weight_bias=*/1.0F, epsilon);
  return cudaGetLastError();
}

// QKRMSNormKernel parallelizes jobs = batch * heads across
// (grid.x blocks, blockDim.y warps); each job normalizes one head row.
template <typename T, uint32_t kVecSize>
cudaError_t LaunchQkTyped(void* query, void* key, const void* query_weight,
                          const void* key_weight, uint64_t tokens, uint64_t query_heads,
                          uint64_t kv_heads, uint64_t head_dim, float epsilon,
                          cudaStream_t stream) {
  constexpr uint32_t kThreads = 128;
  constexpr uint32_t kWarps = kThreads / kWarpSize;
  const uint64_t q_jobs = tokens * query_heads;
  const uint64_t k_jobs = tokens * kv_heads;
  const uint32_t q_blocks =
      static_cast<uint32_t>((q_jobs + kWarps - 1) / kWarps) > 0
          ? static_cast<uint32_t>((q_jobs + kWarps - 1) / kWarps)
          : 1;
  const uint32_t k_blocks =
      static_cast<uint32_t>((k_jobs + kWarps - 1) / kWarps) > 0
          ? static_cast<uint32_t>((k_jobs + kWarps - 1) / kWarps)
          : 1;
  const uint32_t q_stride_n = static_cast<uint32_t>(query_heads * head_dim);
  const uint32_t k_stride_n = static_cast<uint32_t>(kv_heads * head_dim);
  const uint32_t head = static_cast<uint32_t>(head_dim);
  flashinfer::norm::QKRMSNormKernel<kVecSize, T>
      <<<q_blocks, dim3(kWarpSize, kWarps), 0, stream>>>(
          static_cast<T*>(query), static_cast<T*>(const_cast<void*>(query_weight)),
          static_cast<T*>(query), /*d=*/head, /*batch_size=*/static_cast<uint32_t>(tokens),
          /*num_heads=*/static_cast<uint32_t>(query_heads),
          /*stride_input_n=*/q_stride_n, /*stride_input_h=*/head,
          /*stride_output_n=*/q_stride_n, /*stride_output_h=*/head,
          /*weight_bias=*/0.0F, epsilon);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  flashinfer::norm::QKRMSNormKernel<kVecSize, T>
      <<<k_blocks, dim3(kWarpSize, kWarps), 0, stream>>>(
          static_cast<T*>(key), static_cast<T*>(const_cast<void*>(key_weight)),
          static_cast<T*>(key), head, static_cast<uint32_t>(tokens),
          static_cast<uint32_t>(kv_heads), k_stride_n, head, k_stride_n, head, 0.0F, epsilon);
  return cudaGetLastError();
}

template <typename T>
uint32_t VecSizeFor() {
  return 16 / sizeof(T);
}

}  // namespace

cudaError_t LaunchFlashInferFusedAddRmsNorm(const void* input, const void* residual,
                                            const void* weight, uint64_t tokens,
                                            uint64_t hidden, float epsilon, StorageType dtype,
                                            cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  switch (dtype) {
    case StorageType::kFloat32:
      return LaunchFusedAddTyped<float, 4>(input, residual, weight, tokens, hidden, epsilon,
                                           stream);
    case StorageType::kFloat16:
      return LaunchFusedAddTyped<__half, 8>(input, residual, weight, tokens, hidden, epsilon,
                                            stream);
    case StorageType::kBFloat16:
      return LaunchFusedAddTyped<__nv_bfloat16, 8>(input, residual, weight, tokens, hidden,
                                                   epsilon, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t LaunchFlashInferGemmaRmsNorm(const void* input, const void* weight, void* output,
                                         uint64_t tokens, uint64_t hidden, float epsilon,
                                         StorageType dtype, cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  switch (dtype) {
    case StorageType::kFloat32:
      return LaunchGemmaTyped<float, 4>(input, weight, output, tokens, hidden, epsilon, stream);
    case StorageType::kFloat16:
      return LaunchGemmaTyped<__half, 8>(input, weight, output, tokens, hidden, epsilon, stream);
    case StorageType::kBFloat16:
      return LaunchGemmaTyped<__nv_bfloat16, 8>(input, weight, output, tokens, hidden, epsilon,
                                                stream);
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t LaunchFlashInferQkRmsNorm(void* query, void* key, const void* query_weight,
                                      const void* key_weight, uint64_t tokens,
                                      uint64_t query_heads, uint64_t kv_heads,
                                      uint64_t head_dim, float epsilon, StorageType dtype,
                                      cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  switch (dtype) {
    case StorageType::kFloat32:
      return LaunchQkTyped<float, 4>(query, key, query_weight, key_weight, tokens, query_heads,
                                     kv_heads, head_dim, epsilon, stream);
    case StorageType::kFloat16:
      return LaunchQkTyped<__half, 8>(query, key, query_weight, key_weight, tokens, query_heads,
                                      kv_heads, head_dim, epsilon, stream);
    case StorageType::kBFloat16:
      return LaunchQkTyped<__nv_bfloat16, 8>(query, key, query_weight, key_weight, tokens,
                                             query_heads, kv_heads, head_dim, epsilon, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace inferx::kernels::cuda::layernorm
