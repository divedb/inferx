// attn_res (Kimi-style residual-stream mixing): last-resort custom kernel
// (TokenSpeed's CUDA kernel is Blackwell-only; its portable path is Triton).
// One block per token; hidden staged in two streaming passes with block
// reductions for the per-candidate logits, then a weighted combine.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "inferx/kernels/cuda/kernel_utils.cuh"
#include "inferx/kernels/cuda/model_fused_kernels.h"

namespace inferx::kernels::cuda::attnres {
namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxCandidates = 13;  // blocks <= 12 plus layer residual

template <typename T>
__global__ void AttnResKernel(const T* __restrict__ block_residual,
                              T* __restrict__ layer_residual, const T* __restrict__ res_weight,
                              const T* __restrict__ rms_weight,
                              const T* __restrict__ out_norm_weight, uint64_t tokens,
                              uint64_t hidden, uint64_t blocks, float epsilon,
                              float out_norm_epsilon) {
  const uint64_t token = blockIdx.x;
  if (token >= tokens) return;
  const uint32_t count = static_cast<uint32_t>(blocks) + 1;
  const T* candidates[kMaxCandidates];
  for (uint32_t b = 0; b < blocks; ++b) {
    candidates[b] = block_residual + ((static_cast<uint64_t>(b) * tokens) + token) * hidden;
  }
  candidates[blocks] = layer_residual + token * hidden;

  __shared__ float shared_logits[kMaxCandidates];
  __shared__ float shared_sums[kThreads];
  __shared__ float shared_norm[kMaxCandidates];

  // Pass 1: per-candidate rms and logit contributions.
  for (uint32_t n = 0; n < count; ++n) {
    float square_sum = 0.0F;
    float dot = 0.0F;
    for (uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      const float value = Load(candidates[n], h);
      square_sum += value * value;
      dot += value * Load(rms_weight, h) * Load(res_weight, h);
    }
    shared_sums[threadIdx.x] = square_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
      __syncthreads();
    }
    const float inverse_rms =
        rsqrtf(shared_sums[0] / static_cast<float>(hidden) + epsilon);
    shared_norm[n] = inverse_rms;
    __syncthreads();
    shared_sums[threadIdx.x] = dot;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      shared_logits[n] = shared_sums[0] * inverse_rms / sqrtf(static_cast<float>(hidden));
    }
    __syncthreads();
  }

  // Softmax over candidates.
  if (threadIdx.x == 0) {
    float maximum = -INFINITY;
    for (uint32_t n = 0; n < count; ++n) maximum = fmaxf(maximum, shared_logits[n]);
    float total = 0.0F;
    for (uint32_t n = 0; n < count; ++n) {
      shared_logits[n] = __expf(shared_logits[n] - maximum);
      total += shared_logits[n];
    }
    for (uint32_t n = 0; n < count; ++n) shared_logits[n] /= total;
  }
  __syncthreads();

  // Pass 2: weighted combine (+ optional output RMSNorm accumulation).
  __shared__ float shared_square[kThreads];
  float local_square = 0.0F;
  for (uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float mixed = 0.0F;
    for (uint32_t n = 0; n < count; ++n) {
      mixed += shared_logits[n] * Load(candidates[n], h);
    }
    local_square += mixed * mixed;
    // stage into layer_residual temporarily; norm applied next
    Store(layer_residual + token * hidden, h, mixed);
  }
  if (out_norm_weight != nullptr) {
    shared_square[threadIdx.x] = local_square;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) shared_square[threadIdx.x] += shared_square[threadIdx.x + stride];
      __syncthreads();
    }
    const float inverse_rms =
        rsqrtf(shared_square[0] / static_cast<float>(hidden) + out_norm_epsilon);
    for (uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      const float mixed = Load(layer_residual + token * hidden, h);
      Store(layer_residual + token * hidden, h,
            mixed * inverse_rms * Load(out_norm_weight, h));
    }
  }
}

}  // namespace

cudaError_t LaunchAttnRes(const void* block_residual, void* layer_residual,
                          const void* res_weight, const void* rms_weight,
                          const void* out_norm_weight, uint64_t tokens, uint64_t hidden,
                          uint64_t blocks, float epsilon, float out_norm_epsilon,
                          StorageType dtype, cudaStream_t stream) {
  if (tokens == 0 || blocks == 0 || blocks + 1 > kMaxCandidates) return cudaErrorInvalidValue;
  switch (dtype) {
    case StorageType::kFloat32:
      AttnResKernel<float><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const float*>(block_residual), static_cast<float*>(layer_residual),
          static_cast<const float*>(res_weight), static_cast<const float*>(rms_weight),
          static_cast<const float*>(out_norm_weight), tokens, hidden, blocks, epsilon,
          out_norm_epsilon);
      break;
    case StorageType::kFloat16:
      AttnResKernel<__half><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const __half*>(block_residual), static_cast<__half*>(layer_residual),
          static_cast<const __half*>(res_weight), static_cast<const __half*>(rms_weight),
          static_cast<const __half*>(out_norm_weight), tokens, hidden, blocks, epsilon,
          out_norm_epsilon);
      break;
    case StorageType::kBFloat16:
      AttnResKernel<__nv_bfloat16><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(block_residual),
          static_cast<__nv_bfloat16*>(layer_residual),
          static_cast<const __nv_bfloat16*>(res_weight),
          static_cast<const __nv_bfloat16*>(rms_weight),
          static_cast<const __nv_bfloat16*>(out_norm_weight), tokens, hidden, blocks, epsilon,
          out_norm_epsilon);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace inferx::kernels::cuda::attnres
