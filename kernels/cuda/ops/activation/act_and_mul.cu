// Fused-layout activation-and-mul via FlashInfer's device kernel (ADR 0032
// reuse tier 2): the [tokens, 2*d] gate|up convention used by FlashInfer and
// TokenSpeed. Kind dispatch: 0 = silu, 1 = gelu (erf), 2 = gelu (tanh).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "flashinfer/activation.cuh"
#include "inferx/kernels/cuda/activation_kernels.h"

namespace inferx::kernels::cuda::activation {
namespace {

__device__ __forceinline__ float SiluActivation(const float& value) {
  return value / (1.0F + __expf(-value));
}

__device__ __forceinline__ float GeluErfActivation(const float& value) {
  return 0.5F * value * (1.0F + erff(value * 0.70710678118654752440F));
}

__device__ __forceinline__ float GeluTanhActivation(const float& value) {
  const float inner = 0.7978845608028654F * (value + 0.044715F * value * value * value);
  return 0.5F * value * (1.0F + tanhf(inner));
}

template <typename T, float (*Activation)(const float&)>
cudaError_t LaunchTyped(void* output, const void* input, uint64_t tokens, uint64_t half_dim,
                        cudaStream_t stream) {
  const uint32_t blocks = static_cast<uint32_t>(tokens);
  flashinfer::activation::act_and_mul_kernel<T, Activation>
      <<<blocks, 256, 0, stream>>>(static_cast<T*>(output), static_cast<const T*>(input),
                                   static_cast<int>(half_dim));
  return cudaGetLastError();
}

template <typename T>
cudaError_t DispatchKind(void* output, const void* input, uint64_t tokens, uint64_t half_dim,
                         uint32_t kind, cudaStream_t stream) {
  switch (kind) {
    case 0:
      return LaunchTyped<T, SiluActivation>(output, input, tokens, half_dim, stream);
    case 1:
      return LaunchTyped<T, GeluErfActivation>(output, input, tokens, half_dim, stream);
    case 2:
      return LaunchTyped<T, GeluTanhActivation>(output, input, tokens, half_dim, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

cudaError_t LaunchActMulKernel(const void* input, void* output, uint64_t tokens,
                               uint64_t half_dim, uint32_t kind, StorageType dtype,
                               cudaStream_t stream) {
  if (tokens == 0 || half_dim == 0) return cudaSuccess;
  switch (dtype) {
    case StorageType::kFloat32:
      return DispatchKind<float>(output, input, tokens, half_dim, kind, stream);
    case StorageType::kFloat16:
      return DispatchKind<__half>(output, input, tokens, half_dim, kind, stream);
    case StorageType::kBFloat16:
      return DispatchKind<__nv_bfloat16>(output, input, tokens, half_dim, kind, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace inferx::kernels::cuda::activation
