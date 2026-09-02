// RMSNorm via FlashInfer's device kernel (ADR 0032 reuse tier 2).
//
// Instantiates flashinfer::norm::RMSNormKernel ahead of time: no JIT
// modules, no cubin loading, no Python in the closure. The kernel is
// warp-reduced and vectorized (vec_t loads) rather than the scalar
// two-pass reference shape.
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

uint32_t WarpsForHidden(uint64_t hidden, uint32_t vec_size) {
  const uint64_t lanes = (hidden + vec_size - 1) / vec_size;
  uint64_t warps = (lanes + kWarpSize - 1) / kWarpSize;
  if (warps == 0) warps = 1;
  if (warps > kMaxWarps) warps = kMaxWarps;
  return static_cast<uint32_t>(warps);
}

template <typename T, uint32_t kVecSize>
cudaError_t LaunchTyped(const void* input, const void* weight, void* output, uint64_t tokens,
                        uint64_t hidden, float epsilon, cudaStream_t stream) {
  if (hidden % kVecSize != 0) return cudaErrorInvalidValue;
  const uint32_t num_warps = WarpsForHidden(hidden, kVecSize);
  const dim3 grid(static_cast<uint32_t>(tokens));
  const dim3 block(kWarpSize, num_warps);
  const size_t shared_bytes = num_warps * sizeof(float);
  flashinfer::norm::RMSNormKernel<kVecSize, T><<<grid, block, shared_bytes, stream>>>(
      static_cast<T*>(const_cast<void*>(input)), static_cast<T*>(const_cast<void*>(weight)),
      static_cast<T*>(output), static_cast<uint32_t>(hidden), static_cast<uint32_t>(hidden),
      static_cast<uint32_t>(hidden),
      /*weight_bias=*/0.0F, epsilon);
  return cudaGetLastError();
}

}  // namespace

cudaError_t LaunchFlashInferRmsNorm(const void* input, const void* weight, void* output,
                                    uint64_t tokens, uint64_t hidden, float epsilon,
                                    StorageType dtype, cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  switch (dtype) {
    case StorageType::kFloat32:
      return LaunchTyped<float, 4>(input, weight, output, tokens, hidden, epsilon, stream);
    case StorageType::kFloat16:
      return LaunchTyped<__half, 8>(input, weight, output, tokens, hidden, epsilon, stream);
    case StorageType::kBFloat16:
      return LaunchTyped<__nv_bfloat16, 8>(input, weight, output, tokens, hidden, epsilon, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace inferx::kernels::cuda::layernorm
