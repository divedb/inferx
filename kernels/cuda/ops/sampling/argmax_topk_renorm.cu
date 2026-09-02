// Owned sampling kernels: NaN-safe argmax and in-place top-k renorm (no
// external provider at the pin: flashinfer's sampling.cuh carries top-p
// renorm but no top-k renorm or argmax).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "inferx/kernels/cuda/kernel_utils.cuh"
#include "inferx/kernels/cuda/sampling_kernels.h"

namespace inferx::kernels::cuda::sampling {
namespace {

constexpr uint32_t kThreads = 256;
constexpr int kMaxFloatBits = 0x7f800000;

// Argmax: one block per row; NaNs are skipped, ties keep the lowest index,
// an all-NaN row yields -1.
template <typename T>
__global__ void ArgmaxKernel(const T* __restrict__ logits, int32_t* __restrict__ output,
                             uint64_t vocab) {
  const uint64_t row = blockIdx.x;
  const T* row_data = logits + row * vocab;
  float best = -INFINITY;
  int64_t best_index = -1;
  for (uint64_t index = threadIdx.x; index < vocab; index += blockDim.x) {
    const float value = Load(row_data, index);
    if (isnan(value)) continue;
    if (value > best) {
      best = value;
      best_index = static_cast<int64_t>(index);
    }
  }
  __shared__ float shared_values[kThreads];
  __shared__ int64_t shared_indices[kThreads];
  shared_values[threadIdx.x] = best;
  shared_indices[threadIdx.x] = best_index;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const bool replace = shared_values[threadIdx.x + stride] > shared_values[threadIdx.x] ||
                           isnan(shared_values[threadIdx.x]);
      if (replace) {
        shared_values[threadIdx.x] = shared_values[threadIdx.x + stride];
        shared_indices[threadIdx.x] = shared_indices[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[row] = static_cast<int32_t>(shared_indices[0]);
}

// Top-k renorm: one block per row. Pass 1 finds the k-th largest value with
// a binary search over float bit patterns (monotone count predicate); pass 2
// renormalizes entries strictly above the threshold. Equal-to-threshold ties
// can keep slightly more than k entries; the CPU oracle tolerates the same
// slack by comparing distributions after clamping — tests use distinct
// values, where the result is exact.
template <typename T>
__global__ void TopKRenormKernel(T* __restrict__ probs, uint64_t vocab, uint32_t top_k) {
  const uint64_t row = blockIdx.x;
  T* row_data = probs + row * vocab;
  int low = 0;
  int high = kMaxFloatBits;  // positive floats compare as integers
  __shared__ int s_low;
  __shared__ int s_high;
  if (threadIdx.x == 0) {
    s_low = low;
    s_high = high;
  }
  __syncthreads();
  while (true) {
    __shared__ int s_mid;
    __shared__ unsigned int s_count;
    if (threadIdx.x == 0) {
      s_mid = s_low + (s_high - s_low) / 2;
      s_count = 0;
    }
    __syncthreads();
    const int mid = s_mid;
    if (mid == s_low || mid == s_high) break;
    unsigned int local_count = 0;
    for (uint64_t index = threadIdx.x; index < vocab; index += blockDim.x) {
      const float value = Load(row_data, index);
      const int bits = value >= 0.0F ? __float_as_int(value) : 0;
      if (bits >= mid) ++local_count;
    }
    atomicAdd(&s_count, local_count);
    __syncthreads();
    if (threadIdx.x == 0) {
      if (s_count >= top_k) {
        s_low = mid;
      } else {
        s_high = mid;
      }
    }
    __syncthreads();
  }
  const int threshold_bits = s_low;
  float local_sum = 0.0F;
  for (uint64_t index = threadIdx.x; index < vocab; index += blockDim.x) {
    const float value = Load(row_data, index);
    const int bits = value >= 0.0F ? __float_as_int(value) : 0;
    if (bits < threshold_bits)
      Store(row_data, index, 0.0F);
    else
      local_sum += value;
  }
  __shared__ float shared_sum[kThreads];
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    __syncthreads();
  }
  const float total = shared_sum[0];
  for (uint64_t index = threadIdx.x; index < vocab; index += blockDim.x) {
    if (Load(row_data, index) > 0.0F) Store(row_data, index, Load(row_data, index) / total);
  }
}

}  // namespace

cudaError_t LaunchArgmax(const void* logits, uint64_t rows, uint64_t vocab, int32_t* output,
                         StorageType dtype, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  switch (dtype) {
    case StorageType::kFloat32:
      ArgmaxKernel<float><<<static_cast<uint32_t>(rows), kThreads, 0, stream>>>(
          static_cast<const float*>(logits), output, vocab);
      break;
    case StorageType::kFloat16:
      ArgmaxKernel<__half><<<static_cast<uint32_t>(rows), kThreads, 0, stream>>>(
          static_cast<const __half*>(logits), output, vocab);
      break;
    case StorageType::kBFloat16:
      ArgmaxKernel<__nv_bfloat16><<<static_cast<uint32_t>(rows), kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(logits), output, vocab);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t LaunchTopKRenorm(void* probs, uint64_t rows, uint64_t vocab, uint32_t top_k,
                             StorageType dtype, cudaStream_t stream) {
  if (rows == 0 || top_k == 0) return cudaErrorInvalidValue;
  switch (dtype) {
    case StorageType::kFloat32:
      TopKRenormKernel<float><<<static_cast<uint32_t>(rows), kThreads, 0, stream>>>(
          static_cast<float*>(probs), vocab, top_k);
      break;
    case StorageType::kFloat16:
      TopKRenormKernel<__half><<<static_cast<uint32_t>(rows), kThreads, 0, stream>>>(
          static_cast<__half*>(probs), vocab, top_k);
      break;
    case StorageType::kBFloat16:
      TopKRenormKernel<__nv_bfloat16><<<static_cast<uint32_t>(rows), kThreads, 0, stream>>>(
          static_cast<__nv_bfloat16*>(probs), vocab, top_k);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace inferx::kernels::cuda::sampling
