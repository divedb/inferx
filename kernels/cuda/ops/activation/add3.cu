// 3-way residual add: last-resort custom kernel (no external provider at
// the pins; TokenSpeed implements it in Triton).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/activation_kernels.h"
#include "inferx/kernels/cuda/kernel_utils.cuh"

namespace inferx::kernels::cuda::activation {
namespace {

template <typename T>
__global__ void Add3Kernel(const T* __restrict__ a, const T* __restrict__ b,
                           const T* __restrict__ c, T* __restrict__ output,
                           uint64_t elements) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) Store(output, index, Load(a, index) + Load(b, index) + Load(c, index));
}

}  // namespace

cudaError_t LaunchAdd3Kernel(const void* a, const void* b, const void* c, void* output,
                             uint64_t elements, StorageType dtype, cudaStream_t stream) {
  if (elements == 0) return cudaSuccess;
  uint32_t blocks = 0;
  cudaError_t status = CheckedBlocks(elements, &blocks);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      Add3Kernel<float><<<blocks, kKernelThreads, 0, stream>>>(
          static_cast<const float*>(a), static_cast<const float*>(b), static_cast<const float*>(c),
          static_cast<float*>(output), elements);
      break;
    case StorageType::kFloat16:
      Add3Kernel<__half><<<blocks, kKernelThreads, 0, stream>>>(
          static_cast<const __half*>(a), static_cast<const __half*>(b),
          static_cast<const __half*>(c), static_cast<__half*>(output), elements);
      break;
    case StorageType::kBFloat16:
      Add3Kernel<__nv_bfloat16><<<blocks, kKernelThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(a), static_cast<const __nv_bfloat16*>(b),
          static_cast<const __nv_bfloat16*>(c), static_cast<__nv_bfloat16*>(output), elements);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace inferx::kernels::cuda::activation
