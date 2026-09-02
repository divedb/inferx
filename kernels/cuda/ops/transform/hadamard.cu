// Length-128 Hadamard (Sylvester) transform: last-resort custom kernel
// (TokenSpeed wraps an external CUDA wheel plus a Triton fallback; no
// provider library at our pins). One warp per row: each lane computes 4
// output columns; signs come from popcount(i & j) parity.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/kernel_utils.cuh"
#include "inferx/kernels/cuda/transform_kernels.h"

namespace inferx::kernels::cuda::transform {
namespace {

constexpr uint32_t kDim = 128;

template <typename T>
__global__ void Hadamard128Kernel(const T* __restrict__ input,
                                                        T* __restrict__ output, float scale,
                                                        uint64_t rows) {
  const uint64_t row = static_cast<uint64_t>(blockIdx.x) * (blockDim.x / 32) +
                       threadIdx.x / 32;
  if (row >= rows) return;
  const uint32_t lane = threadIdx.x % 32;
  const T* src = input + row * kDim;
  T* dst = output + row * kDim;
  // Each lane owns 4 columns (128 / 32).
  for (uint32_t local = 0; local < 4; ++local) {
    const uint32_t column = lane * 4 + local;
    float sum = 0.0F;
    for (uint32_t source = 0; source < kDim; ++source) {
      const float sign = (__popc(column & source) & 1U) ? -1.0F : 1.0F;
      sum += sign * Load(src, source);
    }
    Store(dst, column, sum * scale);
  }
}

}  // namespace

cudaError_t LaunchHadamard128(const void* input, void* output, uint64_t rows, float scale,
                              StorageType dtype, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  const uint32_t warps_per_block = 8;
  const uint32_t blocks = static_cast<uint32_t>((rows + warps_per_block - 1) / warps_per_block);
  switch (dtype) {
    case StorageType::kFloat32:
      Hadamard128Kernel<float><<<blocks, warps_per_block * 32, 0, stream>>>(
          static_cast<const float*>(input), static_cast<float*>(output), scale, rows);
      break;
    case StorageType::kFloat16:
      Hadamard128Kernel<__half><<<blocks, warps_per_block * 32, 0, stream>>>(
          static_cast<const __half*>(input), static_cast<__half*>(output), scale, rows);
      break;
    case StorageType::kBFloat16:
      Hadamard128Kernel<__nv_bfloat16><<<blocks, warps_per_block * 32, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(input), static_cast<__nv_bfloat16*>(output), scale,
          rows);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace inferx::kernels::cuda::transform
