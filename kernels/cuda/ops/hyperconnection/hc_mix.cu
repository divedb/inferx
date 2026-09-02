// Qwen gated-residual hyperconnection: mix (down-GEMM + SiLU + up-GEMM via
// the CUTLASS dense GEMM shims + fused gating kernel) and combine. Custom
// tier: no external provider; TokenSpeed ships Triton/CuTe versions.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "inferx/kernels/cuda/gemm_kernels.h"
#include "inferx/kernels/cuda/kernel_utils.cuh"
#include "inferx/kernels/cuda/model_fused_kernels.h"

namespace inferx::kernels::cuda::hc {
namespace {

constexpr uint32_t kThreads = 256;

// activated[token, r] = silu(scale * projected[token, r]); projected has
// row stride (rank + hc_count), activated has row stride rank.
__global__ void SiluScaleKernel(const float* __restrict__ projected,
                                float* __restrict__ activated, uint64_t tokens, uint32_t rank,
                                uint32_t hc_count, float scale) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= tokens * rank) return;
  const uint64_t token = index / rank;
  const uint64_t column = index % rank;
  const float value = scale * projected[token * (rank + hc_count) + column];
  activated[index] = value / (1.0F + __expf(-value));
}

// mixed[h] = (1/hc) * sum_b sigmoid(gate[b, h]) * normalized[b, h];
// inject_b = scale * projected[rank + b].
template <typename T>
__global__ void HcMixKernel(const T* __restrict__ normalized, const float* __restrict__ gate,
                            const float* __restrict__ projected, T* __restrict__ mixed,
                            float* __restrict__ inject, uint64_t tokens, uint32_t hc_count,
                            uint32_t hidden, uint32_t rank, float scale) {
  const uint64_t token = blockIdx.x;
  const uint64_t wide = static_cast<uint64_t>(hc_count) * hidden;
  for (uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float sum = 0.0F;
    for (uint32_t b = 0; b < hc_count; ++b) {
      const float gate_value = gate[token * wide + b * hidden + h];
      const float sigmoid_gate = 1.0F / (1.0F + __expf(-gate_value));
      sum += sigmoid_gate * Load(normalized + token * wide + b * hidden, h);
    }
    Store(mixed + token * hidden, h, sum / static_cast<float>(hc_count));
  }
  for (uint32_t b = threadIdx.x; b < hc_count; b += blockDim.x) {
    inject[token * hc_count + b] = scale * projected[token * (rank + hc_count) + rank + b];
  }
}

// residual'_{b,h} += 2 * sigmoid(inject_b) * block_output_h.
template <typename T>
__global__ void HcCombineKernel(const T* __restrict__ block_output, T* __restrict__ residual,
                                const float* __restrict__ inject, uint64_t tokens,
                                uint32_t hc_count, uint32_t hidden) {
  const uint64_t token = blockIdx.x;
  const uint64_t wide = static_cast<uint64_t>(hc_count) * hidden;
  for (uint64_t index = threadIdx.x; index < static_cast<uint64_t>(hc_count) * hidden;
       index += blockDim.x) {
    const uint32_t b = static_cast<uint32_t>(index / hidden);
    const uint64_t h = index % hidden;
    const float scale = 2.0F / (1.0F + __expf(-inject[token * hc_count + b]));
    Store(residual + token * wide + b * hidden + h, h,
          Load(residual + token * wide + b * hidden, h) +
              scale * Load(block_output + token * hidden, h));
  }
}

}  // namespace

uint64_t HcMixWorkspaceBytes(uint64_t tokens, uint32_t rank, uint32_t hc_count,
                             uint32_t hidden) {
  const uint64_t wide = static_cast<uint64_t>(hc_count) * hidden;
  // projected [tokens, rank + hc] + activated [tokens, rank] + gate [tokens, wide]
  return tokens * ((rank + hc_count) + rank + wide) * sizeof(float);
}

cudaError_t LaunchHcMix(const void* normalized, const void* projection_weight,
                        const void* up_weight, void* mixed, void* inject_logits,
                        void* workspace, uint64_t tokens, uint32_t rank, uint32_t hc_count,
                        uint32_t hidden, float projection_scale, StorageType dtype,
                        cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  const uint64_t wide = static_cast<uint64_t>(hc_count) * hidden;
  float* projected = static_cast<float*>(workspace);
  float* activated = projected + tokens * (rank + hc_count);
  float* gate = activated + tokens * rank;
  // projected = normalized @ projection_weight^T  (FP32 accumulate).
  cudaError_t status =
      gemm::LaunchCutlassGemm(normalized, projection_weight, nullptr, projected, tokens, wide,
                              rank + hc_count, 1.0F, 0.0F, dtype, stream);
  if (status != cudaSuccess) return status;
  const uint64_t activated_elements = tokens * rank;
  const uint32_t blocks =
      static_cast<uint32_t>((activated_elements + kThreads - 1) / kThreads);
  SiluScaleKernel<<<blocks, kThreads, 0, stream>>>(projected, activated, tokens, rank, hc_count,
                                                   projection_scale);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  // gate = activated @ up_weight^T.
  status = gemm::LaunchCutlassGemm(activated, up_weight, nullptr, gate, tokens, rank, wide, 1.0F,
                                   0.0F, StorageType::kFloat32, stream);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      HcMixKernel<float><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const float*>(normalized), gate, projected, static_cast<float*>(mixed),
          static_cast<float*>(inject_logits), tokens, hc_count, hidden, rank, projection_scale);
      break;
    case StorageType::kFloat16:
      HcMixKernel<__half><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const __half*>(normalized), gate, projected, static_cast<__half*>(mixed),
          static_cast<float*>(inject_logits), tokens, hc_count, hidden, rank, projection_scale);
      break;
    case StorageType::kBFloat16:
      HcMixKernel<__nv_bfloat16><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(normalized), gate, projected,
          static_cast<__nv_bfloat16*>(mixed), static_cast<float*>(inject_logits), tokens,
          hc_count, hidden, rank, projection_scale);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t LaunchHcCombine(const void* block_output, void* residual, const void* inject_logits,
                            uint64_t tokens, uint32_t hc_count, uint32_t hidden,
                            StorageType dtype, cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  switch (dtype) {
    case StorageType::kFloat32:
      HcCombineKernel<float><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const float*>(block_output), static_cast<float*>(residual),
          static_cast<const float*>(inject_logits), tokens, hc_count, hidden);
      break;
    case StorageType::kFloat16:
      HcCombineKernel<__half><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const __half*>(block_output), static_cast<__half*>(residual),
          static_cast<const float*>(inject_logits), tokens, hc_count, hidden);
      break;
    case StorageType::kBFloat16:
      HcCombineKernel<__nv_bfloat16><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(block_output),
          static_cast<__nv_bfloat16*>(residual), static_cast<const float*>(inject_logits),
          tokens, hc_count, hidden);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace inferx::kernels::cuda::hc
