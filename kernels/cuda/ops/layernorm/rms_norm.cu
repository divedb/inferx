#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/kernel_utils.cuh"
#include "inferx/kernels/cuda/layernorm_kernels.h"

namespace inferx::kernels::cuda::layernorm {
namespace {

template <typename T>
__global__ void RmsNormKernel(const T* input, const T* weight, T* output, uint64_t tokens,
                              uint64_t hidden, float epsilon) {
  const uint64_t token = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  float sum = 0.0F;
  for (uint64_t index = 0; index < hidden; ++index) {
    const float value = Load(input, token * hidden + index);
    sum += value * value;
  }
  const float inverse_rms = rsqrtf(sum / static_cast<float>(hidden) + epsilon);
  for (uint64_t index = 0; index < hidden; ++index) {
    Store(output, token * hidden + index,
          Load(input, token * hidden + index) * inverse_rms * Load(weight, index));
  }
}

template <typename T>
void LaunchRmsNormTyped(uint32_t blocks, const void* input, const void* weight, void* output,
                        uint64_t tokens, uint64_t hidden, float epsilon, cudaStream_t stream) {
  RmsNormKernel<T><<<blocks, kKernelThreads, 0, stream>>>(
      static_cast<const T*>(input), static_cast<const T*>(weight), static_cast<T*>(output), tokens,
      hidden, epsilon);
}

}  // namespace

cudaError_t LaunchRmsNormKernel(const void* input, const void* weight, void* output,
                                uint64_t tokens, uint64_t hidden, float epsilon, StorageType dtype,
                                cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  uint32_t blocks = 0;
  cudaError_t status = CheckedBlocks(tokens, &blocks);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      LaunchRmsNormTyped<float>(blocks, input, weight, output, tokens, hidden, epsilon, stream);
      break;
    case StorageType::kFloat16:
      LaunchRmsNormTyped<__half>(blocks, input, weight, output, tokens, hidden, epsilon, stream);
      break;
    case StorageType::kBFloat16:
      LaunchRmsNormTyped<__nv_bfloat16>(blocks, input, weight, output, tokens, hidden, epsilon,
                                        stream);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

}  // namespace inferx::kernels::cuda::layernorm
