// Matrix hyperconnection (mHC): pre (prenorm GEMM via the CUTLASS shim +
// fused sigmoid/Sinkhorn mixing kernel) and post. Custom tier.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "inferx/kernels/cuda/gemm_kernels.h"
#include "inferx/kernels/cuda/kernel_utils.cuh"
#include "inferx/kernels/cuda/model_fused_kernels.h"

namespace inferx::kernels::cuda::mhc {
namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxStreams = 8;

template <typename T>
__global__ void MhcPreKernel(const T* __restrict__ residual, const float* __restrict__ mix,
                             const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
                             T* __restrict__ layer_input, float* __restrict__ post,
                             float* __restrict__ comb, uint64_t tokens, uint32_t streams,
                             uint32_t hidden, float rms_eps, float hc_eps,
                             uint32_t sinkhorn_iters) {
  const uint64_t token = blockIdx.x;
  const uint64_t wide = static_cast<uint64_t>(streams) * hidden;
  const uint32_t mix_rows = 2 * streams + streams * streams;
  const float* mix_row = mix + token * mix_rows;
  const T* residual_row = residual + token * wide;

  __shared__ float shared_sums[kThreads];
  float square_sum = 0.0F;
  for (uint64_t index = threadIdx.x; index < wide; index += blockDim.x) {
    const float value = Load(residual_row, index);
    square_sum += value * value;
  }
  shared_sums[threadIdx.x] = square_sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
    __syncthreads();
  }
  const float rms = rsqrtf(shared_sums[0] / static_cast<float>(wide) + rms_eps);

  __shared__ float shared_pre[kMaxStreams];
  __shared__ float shared_comb[kMaxStreams * kMaxStreams];
  for (uint32_t i = threadIdx.x; i < streams; i += blockDim.x) {
    const float gate = 1.0F / (1.0F + __expf(-(mix_row[i] * rms * hc_scale[0] + hc_base[i])));
    shared_pre[i] = gate + hc_eps;
    post[token * streams + i] =
        2.0F / (1.0F + __expf(-(mix_row[streams + i] * rms * hc_scale[1] + hc_base[streams + i])));
  }
  __syncthreads();

  // comb: raw scores -> row softmax -> sinkhorn normalize/normalize.
  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < streams; ++i) {
      float maximum = -INFINITY;
      for (uint32_t j = 0; j < streams; ++j) {
        const float raw = mix_row[2 * streams + i * streams + j] * rms * hc_scale[2] +
                          hc_base[2 * streams + i * streams + j];
        shared_comb[i * streams + j] = raw;
        maximum = fmaxf(maximum, raw);
      }
      float total = 0.0F;
      for (uint32_t j = 0; j < streams; ++j) {
        shared_comb[i * streams + j] = __expf(shared_comb[i * streams + j] - maximum);
        total += shared_comb[i * streams + j];
      }
      for (uint32_t j = 0; j < streams; ++j) shared_comb[i * streams + j] /= total;
    }
    for (uint32_t iteration = 0; iteration < sinkhorn_iters; ++iteration) {
      for (uint32_t i = 0; i < streams; ++i) {
        float row_sum = 0.0F;
        for (uint32_t j = 0; j < streams; ++j) row_sum += shared_comb[i * streams + j];
        for (uint32_t j = 0; j < streams; ++j) {
          shared_comb[i * streams + j] = shared_comb[i * streams + j] / (row_sum + hc_eps);
        }
      }
      for (uint32_t j = 0; j < streams; ++j) {
        float column_sum = 0.0F;
        for (uint32_t i = 0; i < streams; ++i) column_sum += shared_comb[i * streams + j];
        for (uint32_t i = 0; i < streams; ++i) {
          shared_comb[i * streams + j] = shared_comb[i * streams + j] / (column_sum + hc_eps);
        }
      }
    }
    for (uint32_t i = 0; i < streams; ++i) {
      for (uint32_t j = 0; j < streams; ++j) {
        comb[token * streams * streams + i * streams + j] = shared_comb[i * streams + j];
      }
    }
  }
  __syncthreads();

  // layer_input[h] = sum_i pre_i * residual_i[h].
  for (uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float sum = 0.0F;
    for (uint32_t i = 0; i < streams; ++i) {
      sum += shared_pre[i] * Load(residual_row + i * hidden, h);
    }
    Store(layer_input + token * hidden, h, sum);
  }
}

// One block per (token, output stream j): writes row j only, so reads of
// all stream rows (including the not-yet-written j) are race-free.
template <typename T>
__global__ void MhcPostKernel(const T* __restrict__ hidden_states, T* __restrict__ residual,
                              const float* __restrict__ post, const float* __restrict__ comb,
                              uint32_t streams, uint32_t hidden) {
  const uint64_t token = blockIdx.x;
  const uint32_t j = blockIdx.y;
  const uint64_t wide = static_cast<uint64_t>(streams) * hidden;
  for (uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float sum = post[token * streams + j] * Load(hidden_states + token * hidden, h);
    for (uint32_t i = 0; i < streams; ++i) {
      sum += comb[token * streams * streams + i * streams + j] *
             Load(residual + token * wide + i * hidden, h);
    }
    Store(residual + token * wide + j * hidden, h, sum);
  }
}

}  // namespace

uint64_t MhcPreWorkspaceBytes(uint64_t tokens, uint32_t streams) {
  return tokens * (2ULL * streams + static_cast<uint64_t>(streams) * streams) * sizeof(float);
}

cudaError_t LaunchMhcPre(const void* residual, const void* fn, const void* hc_scale,
                         const void* hc_base, void* mix_workspace, void* layer_input, void* post,
                         void* comb, uint64_t tokens, uint32_t streams, uint32_t hidden,
                         float rms_eps, float hc_eps, uint32_t sinkhorn_iters, StorageType dtype,
                         cudaStream_t stream) {
  if (tokens == 0 || streams == 0 || streams > kMaxStreams) return cudaErrorInvalidValue;
  const uint64_t wide = static_cast<uint64_t>(streams) * hidden;
  const uint64_t mix_rows = 2ULL * streams + static_cast<uint64_t>(streams) * streams;
  // mix = residual_flat @ fn^T (FP32 accumulate into the workspace).
  cudaError_t status = gemm::LaunchCutlassGemm(residual, fn, nullptr, mix_workspace, tokens, wide,
                                               mix_rows, 1.0F, 0.0F, dtype, stream);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      MhcPreKernel<float><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const float*>(residual), static_cast<const float*>(mix_workspace),
          static_cast<const float*>(hc_scale), static_cast<const float*>(hc_base),
          static_cast<float*>(layer_input), static_cast<float*>(post), static_cast<float*>(comb),
          tokens, streams, hidden, rms_eps, hc_eps, sinkhorn_iters);
      break;
    case StorageType::kFloat16:
      MhcPreKernel<__half><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const __half*>(residual), static_cast<const float*>(mix_workspace),
          static_cast<const float*>(hc_scale), static_cast<const float*>(hc_base),
          static_cast<__half*>(layer_input), static_cast<float*>(post), static_cast<float*>(comb),
          tokens, streams, hidden, rms_eps, hc_eps, sinkhorn_iters);
      break;
    case StorageType::kBFloat16:
      MhcPreKernel<__nv_bfloat16><<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(residual), static_cast<const float*>(mix_workspace),
          static_cast<const float*>(hc_scale), static_cast<const float*>(hc_base),
          static_cast<__nv_bfloat16*>(layer_input), static_cast<float*>(post),
          static_cast<float*>(comb), tokens, streams, hidden, rms_eps, hc_eps, sinkhorn_iters);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t LaunchMhcPost(const void* hidden_states, void* residual, const void* post,
                          const void* comb, uint64_t tokens, uint32_t streams, uint32_t hidden,
                          StorageType dtype, cudaStream_t stream) {
  if (tokens == 0 || streams == 0) return cudaSuccess;
  switch (dtype) {
    case StorageType::kFloat32:
      MhcPostKernel<float><<<dim3(static_cast<uint32_t>(tokens), streams), kThreads, 0, stream>>>(
          static_cast<const float*>(hidden_states), static_cast<float*>(residual),
          static_cast<const float*>(post), static_cast<const float*>(comb), streams, hidden);
      break;
    case StorageType::kFloat16:
      MhcPostKernel<__half><<<dim3(static_cast<uint32_t>(tokens), streams), kThreads, 0, stream>>>(
          static_cast<const __half*>(hidden_states), static_cast<__half*>(residual),
          static_cast<const float*>(post), static_cast<const float*>(comb), streams, hidden);
      break;
    case StorageType::kBFloat16:
      MhcPostKernel<__nv_bfloat16>
          <<<dim3(static_cast<uint32_t>(tokens), streams), kThreads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(hidden_states),
              static_cast<__nv_bfloat16*>(residual), static_cast<const float*>(post),
              static_cast<const float*>(comb), streams, hidden);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace inferx::kernels::cuda::mhc
