// Residual add: last-resort custom kernel (ADR 0032 provider audit).
//
// Neither provider exposes a plain add: hpc-ops fuses residual into
// allreduce+rmsnorm (collective path), flashinfer fuses it into
// FusedAddRMSNorm. Minimal correctness kernel by design.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/activation_kernels.h"
#include "inferx/kernels/cuda/kernel_utils.cuh"

namespace inferx::kernels::cuda::activation {
namespace {

template <typename T>
__global__ void ResidualKernel(const T* __restrict__ left, const T* __restrict__ right,
                               T* __restrict__ output, uint64_t elements) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) Store(output, index, Load(left, index) + Load(right, index));
}

}  // namespace

cudaError_t LaunchResidualKernel(const void* left, const void* right, void* output,
                                 uint64_t elements, StorageType dtype, cudaStream_t stream) {
  if (elements == 0) return cudaSuccess;
  uint32_t blocks = 0;
  cudaError_t status = CheckedBlocks(elements, &blocks);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      ResidualKernel<float><<<blocks, kKernelThreads, 0, stream>>>(
          static_cast<const float*>(left), static_cast<const float*>(right),
          static_cast<float*>(output), elements);
      break;
    case StorageType::kFloat16:
      ResidualKernel<__half><<<blocks, kKernelThreads, 0, stream>>>(
          static_cast<const __half*>(left), static_cast<const __half*>(right),
          static_cast<__half*>(output), elements);
      break;
    case StorageType::kBFloat16:
      ResidualKernel<__nv_bfloat16><<<blocks, kKernelThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(left), static_cast<const __nv_bfloat16*>(right),
          static_cast<__nv_bfloat16*>(output), elements);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

}  // namespace inferx::kernels::cuda::activation
