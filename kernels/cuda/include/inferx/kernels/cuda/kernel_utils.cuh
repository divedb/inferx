// Shared helpers for the owned CUDA kernels under kernels/cuda/ops/*.
// Device-only translation units include this header; it carries no Abseil or
// InferX host types (ADR 0030 policy).
#ifndef INFERX_KERNELS_CUDA_KERNEL_UTILS_CUH_
#define INFERX_KERNELS_CUDA_KERNEL_UTILS_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace inferx::kernels::cuda {

constexpr uint32_t kKernelThreads = 128;
constexpr uint64_t kKernelMaximumGridX = 2'147'483'647ULL;

template <typename T>
__device__ float Load(const T* values, uint64_t index) {
  return static_cast<float>(values[index]);
}

template <>
__device__ inline float Load<__half>(const __half* values, uint64_t index) {
  return __half2float(values[index]);
}

template <>
__device__ inline float Load<__nv_bfloat16>(const __nv_bfloat16* values, uint64_t index) {
  return __bfloat162float(values[index]);
}

template <typename T>
__device__ void Store(T* values, uint64_t index, float value) {
  values[index] = static_cast<T>(value);
}

template <>
__device__ inline void Store<__half>(__half* values, uint64_t index, float value) {
  values[index] = __float2half_rn(value);
}

template <>
__device__ inline void Store<__nv_bfloat16>(__nv_bfloat16* values, uint64_t index, float value) {
  values[index] = __float2bfloat16_rn(value);
}

inline cudaError_t CheckedWork(uint64_t left, uint64_t right, uint64_t* output) {
  if (left != 0 && right > UINT64_MAX / left) return cudaErrorInvalidValue;
  *output = left * right;
  return cudaSuccess;
}

inline cudaError_t CheckedBlocks(uint64_t work, uint32_t* blocks) {
  if (work == 0) {
    *blocks = 0;
    return cudaSuccess;
  }
  const uint64_t count = (work - 1) / kKernelThreads + 1;
  if (count > kKernelMaximumGridX) return cudaErrorInvalidConfiguration;
  *blocks = static_cast<uint32_t>(count);
  return cudaSuccess;
}

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_KERNEL_UTILS_CUH_
