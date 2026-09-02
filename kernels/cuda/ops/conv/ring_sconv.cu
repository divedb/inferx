// Depthwise causal short-FIR convolution with an in-place ring cache
// (TokenSpeed inkling_ring_sconv; Triton there, custom here). One warp per
// token; the warp processes `width` channels with its lanes. window is tiny
// (typically 4), so the per-channel tap loop is a short register loop.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/kernel_utils.cuh"
#include "inferx/kernels/cuda/model_fused_kernels.h"

namespace inferx::kernels::cuda::conv {
namespace {

constexpr uint32_t kMaxWindow = 16;
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpsPerBlock = 8;

template <typename T>
__global__ void RingSconvKernel(const T* __restrict__ x, const T* __restrict__ weight,
                                T* __restrict__ conv_cache, const int32_t* __restrict__ seq_prefix,
                                const int32_t* __restrict__ cache_indices, T* __restrict__ output,
                                uint64_t width, uint32_t window, uint64_t ring) {
  const uint64_t token_global = static_cast<uint64_t>(blockIdx.x) * kWarpsPerBlock +
                                threadIdx.x / kWarpSize;
  if (token_global >= gridDim.y) return;
  const uint32_t lane = threadIdx.x % kWarpSize;
  // gridDim.y carries the total token count (set via the launch below).
  const int32_t total_tokens = static_cast<int32_t>(gridDim.y);
  (void)total_tokens;
  // locate the sequence containing this token
  int32_t sequence = -1;
  int32_t local_step = -1;
  for (uint32_t candidate = 0; candidate < gridDim.z; ++candidate) {
    if (token_global >= static_cast<uint64_t>(seq_prefix[candidate]) &&
        token_global < static_cast<uint64_t>(seq_prefix[candidate + 1])) {
      sequence = static_cast<int32_t>(candidate);
      local_step = static_cast<int32_t>(token_global) - seq_prefix[candidate];
      break;
    }
  }
  if (sequence < 0) return;
  const int32_t slot = cache_indices[sequence];
  T* ring_cache = conv_cache + static_cast<uint64_t>(slot) * ring * width;
  for (uint64_t channel = lane; channel < width; channel += kWarpSize) {
    float sum =
        Load(weight, channel * window + (window - 1)) * Load(x, token_global * width + channel);
    for (uint32_t lag = 1; lag < window; ++lag) {
      const int32_t history_step = local_step - static_cast<int32_t>(lag);
      const float tap = history_step >= 0
                            ? Load(ring_cache,
                                   (static_cast<uint64_t>(history_step) % ring) * width + channel)
                            : 0.0F;
      sum += Load(weight, channel * window + (window - 1 - lag)) * tap;
    }
    Store(output, token_global * width + channel, sum);
  }
  __syncwarp();
  for (uint64_t channel = lane; channel < width; channel += kWarpSize) {
    Store(ring_cache, (static_cast<uint64_t>(local_step) % ring) * width + channel,
          Load(x, token_global * width + channel));
  }
}

}  // namespace

cudaError_t LaunchRingSconv(const void* x, const void* weight, void* conv_cache,
                            const int32_t* seq_prefix, const int32_t* cache_indices, void* output,
                            uint64_t tokens, uint64_t width, uint32_t window, uint64_t ring,
                            uint64_t batch, StorageType dtype, cudaStream_t stream) {
  if (tokens == 0 || window == 0 || window > kMaxWindow || ring < window) {
    return cudaErrorInvalidValue;
  }
  const uint32_t blocks = static_cast<uint32_t>((tokens + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const dim3 grid(blocks, static_cast<uint32_t>(tokens), static_cast<uint32_t>(batch));
  switch (dtype) {
    case StorageType::kFloat32:
      RingSconvKernel<float><<<grid, kWarpsPerBlock * kWarpSize, 0, stream>>>(
          static_cast<const float*>(x), static_cast<const float*>(weight),
          static_cast<float*>(conv_cache), seq_prefix, cache_indices,
          static_cast<float*>(output), width, window, ring);
      break;
    case StorageType::kFloat16:
      RingSconvKernel<__half><<<grid, kWarpsPerBlock * kWarpSize, 0, stream>>>(
          static_cast<const __half*>(x), static_cast<const __half*>(weight),
          static_cast<__half*>(conv_cache), seq_prefix, cache_indices,
          static_cast<__half*>(output), width, window, ring);
      break;
    case StorageType::kBFloat16:
      RingSconvKernel<__nv_bfloat16><<<grid, kWarpsPerBlock * kWarpSize, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(weight),
          static_cast<__nv_bfloat16*>(conv_cache), seq_prefix, cache_indices,
          static_cast<__nv_bfloat16*>(output), width, window, ring);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace inferx::kernels::cuda::conv
