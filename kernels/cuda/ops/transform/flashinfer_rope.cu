// RoPE via FlashInfer's device kernel (ADR 0032 reuse tier 2).
//
// Instantiates flashinfer::BatchQKApplyRotaryPosIdsKernel ahead of time: no
// JIT modules, no cubin loading. The PosIds variant takes per-token position
// IDs, mapping directly onto the InferX RopeRequest; non-interleave mode is
// the NeoX half-split rotation the CPU oracle defines. head_dim is a
// compile-time parameter, so the common dims are instantiated and anything
// else is rejected (Unimplemented at the provider layer).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "flashinfer/pos_enc.cuh"
#include "inferx/kernels/cuda/transform_kernels.h"

namespace inferx::kernels::cuda::transform {
namespace {

template <typename T, uint32_t kHeadDim, uint32_t kVecSize>
cudaError_t LaunchTyped(const void* query, const void* key, const int32_t* positions,
                        void* query_output, void* key_output, uint64_t tokens, uint64_t query_heads,
                        uint64_t kv_heads, float theta, cudaStream_t stream) {
  constexpr uint32_t bdx = kHeadDim / kVecSize;
  const uint64_t bdy64 = 256 / bdx;
  const uint32_t bdy = bdy64 == 0 ? 1 : static_cast<uint32_t>(bdy64);
  const uint32_t blocks = static_cast<uint32_t>((tokens + bdy - 1) / bdy) > 0
                              ? static_cast<uint32_t>((tokens + bdy - 1) / bdy)
                              : 1;
  const dim3 grid(blocks);
  const dim3 block(bdx, bdy);
  flashinfer::BatchQKApplyRotaryPosIdsKernel</*interleave=*/false, kHeadDim, kVecSize, bdx, T,
                                             int32_t><<<grid, block, 0, stream>>>(
      static_cast<T*>(const_cast<void*>(query)), static_cast<T*>(const_cast<void*>(key)),
      static_cast<T*>(query_output), static_cast<T*>(key_output), const_cast<int32_t*>(positions),
      static_cast<uint32_t>(tokens), static_cast<uint32_t>(query_heads),
      static_cast<uint32_t>(kv_heads),
      /*rotary_dim=*/kHeadDim,
      /*q_stride_n=*/query_heads * kHeadDim,
      /*q_stride_h=*/kHeadDim,
      /*k_stride_n=*/kv_heads * kHeadDim,
      /*k_stride_h=*/kHeadDim,
      /*q_rope_stride_n=*/query_heads * kHeadDim,
      /*q_rope_stride_h=*/kHeadDim,
      /*k_rope_stride_n=*/kv_heads * kHeadDim,
      /*k_rope_stride_h=*/kHeadDim,
      /*smooth_a=*/0.0F, /*smooth_b=*/1.0F, /*rope_rcp_scale=*/1.0F,
      /*rope_rcp_theta=*/1.0F / theta);
  return cudaPeekAtLastError();
}

template <typename T, uint32_t kHeadDim>
cudaError_t LaunchForHeadDim(const void* query, const void* key, const int32_t* positions,
                             void* query_output, void* key_output, uint64_t tokens,
                             uint64_t query_heads, uint64_t kv_heads, uint64_t head_dimension,
                             float theta, cudaStream_t stream) {
  constexpr uint32_t kVecSize = 16 / sizeof(T);
  if (head_dimension != kHeadDim || kHeadDim % (2 * kVecSize) != 0) return cudaErrorInvalidValue;
  return LaunchTyped<T, kHeadDim, kVecSize>(query, key, positions, query_output, key_output, tokens,
                                            query_heads, kv_heads, theta, stream);
}

template <typename T>
cudaError_t DispatchHeadDim(const void* query, const void* key, const int32_t* positions,
                            void* query_output, void* key_output, uint64_t tokens,
                            uint64_t query_heads, uint64_t kv_heads, uint64_t head_dimension,
                            float theta, cudaStream_t stream) {
  switch (head_dimension) {
    case 32:
      return LaunchForHeadDim<T, 32>(query, key, positions, query_output, key_output, tokens,
                                     query_heads, kv_heads, head_dimension, theta, stream);
    case 64:
      return LaunchForHeadDim<T, 64>(query, key, positions, query_output, key_output, tokens,
                                     query_heads, kv_heads, head_dimension, theta, stream);
    case 128:
      return LaunchForHeadDim<T, 128>(query, key, positions, query_output, key_output, tokens,
                                      query_heads, kv_heads, head_dimension, theta, stream);
    case 256:
      return LaunchForHeadDim<T, 256>(query, key, positions, query_output, key_output, tokens,
                                      query_heads, kv_heads, head_dimension, theta, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

cudaError_t LaunchFlashInferRope(const void* query, const void* key, const int32_t* positions,
                                 void* query_output, void* key_output, uint64_t tokens,
                                 uint64_t query_heads, uint64_t kv_heads, uint64_t head_dimension,
                                 float theta, StorageType dtype, cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  switch (dtype) {
    case StorageType::kFloat32:
      return DispatchHeadDim<float>(query, key, positions, query_output, key_output, tokens,
                                    query_heads, kv_heads, head_dimension, theta, stream);
    case StorageType::kFloat16:
      return DispatchHeadDim<__half>(query, key, positions, query_output, key_output, tokens,
                                     query_heads, kv_heads, head_dimension, theta, stream);
    case StorageType::kBFloat16:
      return DispatchHeadDim<__nv_bfloat16>(query, key, positions, query_output, key_output, tokens,
                                            query_heads, kv_heads, head_dimension, theta, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace inferx::kernels::cuda::transform
