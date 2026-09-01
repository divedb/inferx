// SiluMultiply (SwiGlu): last-resort custom kernel (ADR 0032 provider audit).
//
// No external provider matches the contract at the pinned revisions: hpc-ops
// activations are FP8-quantization fusions, flashinfer's act_and_mul_kernel
// requires a fused [tokens, 2*d] gate|up buffer while the InferX contract
// carries separate gate/up tensors. Minimal correctness kernel by design.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "inferx/kernels/cuda/activation_kernels.h"
#include "inferx/kernels/cuda/kernel_utils.cuh"

namespace inferx::kernels::cuda::activation {
namespace {

template <typename T>
__global__ void SwiGluKernel(const T* __restrict__ gate, const T* __restrict__ up,
                             T* __restrict__ output, uint64_t elements) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float value = Load(gate, index);
  Store(output, index, (value / (1.0F + expf(-value))) * Load(up, index));
}

}  // namespace

cudaError_t LaunchSwiGluKernel(const void* gate, const void* up, void* output, uint64_t elements,
                               StorageType dtype, cudaStream_t stream) {
  if (elements == 0) return cudaSuccess;
  uint32_t blocks = 0;
  cudaError_t status = CheckedBlocks(elements, &blocks);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      SwiGluKernel<float><<<blocks, kKernelThreads, 0, stream>>>(
          static_cast<const float*>(gate), static_cast<const float*>(up),
          static_cast<float*>(output), elements);
      break;
    case StorageType::kFloat16:
      SwiGluKernel<__half><<<blocks, kKernelThreads, 0, stream>>>(
          static_cast<const __half*>(gate), static_cast<const __half*>(up),
          static_cast<__half*>(output), elements);
      break;
    case StorageType::kBFloat16:
      SwiGluKernel<__nv_bfloat16><<<blocks, kKernelThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(gate), static_cast<const __nv_bfloat16*>(up),
          static_cast<__nv_bfloat16*>(output), elements);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

}  // namespace inferx::kernels::cuda::activation
