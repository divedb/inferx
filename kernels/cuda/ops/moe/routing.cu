// Owned MoE routing kernels: softmax top-k and sigmoid+bias top-k (the
// pinned flashinfer exposes no standalone routing kernel; TokenSpeed ships
// Triton versions). One block per token; experts staged in shared memory;
// selection via k passes over the shared row (k is small in practice).
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "inferx/kernels/cuda/quantization_kernels.h"

namespace inferx::kernels::cuda::moe {
namespace {

constexpr uint32_t kMaxExpertsInShared = 4096;
constexpr uint32_t kThreads = 256;

__device__ __forceinline__ float ExpfStable(float value) { return __expf(value); }

template <bool kSigmoid>
__global__ void RoutingTopKKernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                  float* __restrict__ weights, int32_t* __restrict__ ids,
                                  uint64_t experts, uint32_t top_k, float routed_scale,
                                  bool renormalize) {
  extern __shared__ float shared_logits[];
  const uint64_t token = blockIdx.x;
  const float* row = logits + token * experts;
  for (uint64_t expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    shared_logits[expert] =
        kSigmoid ? row[expert] + (bias != nullptr ? bias[expert] : 0.0F) : row[expert];
  }
  __syncthreads();
  __shared__ float s_denominator;
  if (!kSigmoid) {
    // Single thread: the in-place exponentiation must not race with other
    // threads still reading the raw logits.
    if (threadIdx.x == 0) {
      float maximum = -INFINITY;
      for (uint64_t expert = 0; expert < experts; ++expert) {
        maximum = fmaxf(maximum, shared_logits[expert]);
      }
      float denominator = 0.0F;
      for (uint64_t expert = 0; expert < experts; ++expert) {
        shared_logits[expert] = ExpfStable(shared_logits[expert] - maximum);
        denominator += shared_logits[expert];
      }
      s_denominator = denominator;
    }
    __syncthreads();
  }
  __shared__ float shared_sum;
  if (threadIdx.x == 0) shared_sum = 0.0F;
  __syncthreads();
  for (uint32_t selection = 0; selection < top_k; ++selection) {
    // single thread selects the argmax (experts <= 4096 keeps this cheap)
    if (threadIdx.x == 0) {
      float best = -INFINITY;
      int32_t best_expert = -1;
      for (uint64_t expert = 0; expert < experts; ++expert) {
        if (shared_logits[expert] > best) {
          best = shared_logits[expert];
          best_expert = static_cast<int32_t>(expert);
        }
      }
      float weight;
      if (kSigmoid) {
        const float unbiased = row[best_expert];
        weight = routed_scale / (1.0F + __expf(-unbiased));
      } else {
        weight = best / s_denominator;
      }
      weights[token * top_k + selection] = weight;
      ids[token * top_k + selection] = best_expert;
      shared_logits[best_expert] = -INFINITY;
      if (renormalize) atomicAdd(&shared_sum, weight);
    }
    __syncthreads();
  }
  if (renormalize && threadIdx.x == 0 && shared_sum > 0.0F) {
    for (uint32_t selection = 0; selection < top_k; ++selection) {
      weights[token * top_k + selection] /= shared_sum;
    }
  }
}

}  // namespace

cudaError_t LaunchSoftmaxTopK(const void* logits, void* weights, int32_t* ids, uint64_t tokens,
                              uint64_t experts, uint32_t top_k, bool renormalize,
                              cudaStream_t stream) {
  if (tokens == 0 || top_k == 0 || experts > kMaxExpertsInShared) return cudaErrorInvalidValue;
  RoutingTopKKernel<false>
      <<<static_cast<uint32_t>(tokens), kThreads, experts * sizeof(float), stream>>>(
          static_cast<const float*>(logits), nullptr, static_cast<float*>(weights), ids, experts,
          top_k, 1.0F, renormalize);
  return cudaGetLastError();
}

cudaError_t LaunchSigmoidBiasTopK(const void* logits, const void* bias, void* weights, int32_t* ids,
                                  uint64_t tokens, uint64_t experts, uint32_t top_k,
                                  float routed_scaling_factor, bool renormalize,
                                  cudaStream_t stream) {
  if (tokens == 0 || top_k == 0 || experts > kMaxExpertsInShared) return cudaErrorInvalidValue;
  RoutingTopKKernel<true>
      <<<static_cast<uint32_t>(tokens), kThreads, experts * sizeof(float), stream>>>(
          static_cast<const float*>(logits), static_cast<const float*>(bias),
          static_cast<float*>(weights), ids, experts, top_k, routed_scaling_factor, renormalize);
  return cudaGetLastError();
}

}  // namespace inferx::kernels::cuda::moe
