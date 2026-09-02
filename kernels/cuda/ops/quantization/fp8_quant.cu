// Owned FP8 e4m3 quantization: per tensor / per token / per token-group.
// No external provider at the pins (hpc-ops quant kernels are fused
// activation variants; the pinned flashinfer quantization.cuh carries only
// pack-bits kernels).
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "inferx/kernels/cuda/quantization_kernels.h"

namespace inferx::kernels::cuda::quant {
namespace {

constexpr uint32_t kThreads = 256;
constexpr float kFp8Max = 448.0F;

// One block per token row: begin is derived from blockIdx.x.
__global__ void Fp8QuantTokenKernel(const float* __restrict__ input,
                                    __nv_fp8_e4m3* __restrict__ output,
                                    float* __restrict__ scales, uint64_t dim) {
  const uint64_t begin = static_cast<uint64_t>(blockIdx.x) * dim;
  const uint64_t count = dim;
  float local_amax = 0.0F;
  for (uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
    local_amax = fmaxf(local_amax, fabsf(input[begin + index]));
  }
  __shared__ float shared_amax[kThreads];
  shared_amax[threadIdx.x] = local_amax;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_amax[threadIdx.x] = fmaxf(shared_amax[threadIdx.x], shared_amax[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  __shared__ float shared_scale;
  if (threadIdx.x == 0) {
    shared_scale = shared_amax[0] > 0.0F ? shared_amax[0] / kFp8Max : 1.0F;
    scales[blockIdx.x] = shared_scale;
  }
  __syncthreads();
  const float scale = shared_scale;
  for (uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
    const float value = input[begin + index];
    output[begin + index] = __nv_fp8_e4m3(fminf(fmaxf(value / scale, -kFp8Max), kFp8Max));
  }
}

// One block for the whole tensor (per-tensor granularity).
__global__ void Fp8QuantTensorKernel(const float* __restrict__ input,
                                     __nv_fp8_e4m3* __restrict__ output,
                                     float* __restrict__ scales, uint64_t count) {
  float local_amax = 0.0F;
  for (uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
    local_amax = fmaxf(local_amax, fabsf(input[index]));
  }
  __shared__ float shared_amax[kThreads];
  shared_amax[threadIdx.x] = local_amax;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_amax[threadIdx.x] = fmaxf(shared_amax[threadIdx.x], shared_amax[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  __shared__ float shared_scale;
  if (threadIdx.x == 0) {
    shared_scale = shared_amax[0] > 0.0F ? shared_amax[0] / kFp8Max : 1.0F;
    scales[0] = shared_scale;
  }
  __syncthreads();
  const float scale = shared_scale;
  for (uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
    const float value = input[index];
    output[index] = __nv_fp8_e4m3(fminf(fmaxf(value / scale, -kFp8Max), kFp8Max));
  }
}

// One block per (token, group).
__global__ void Fp8QuantGroupKernel(const float* __restrict__ input,
                                    __nv_fp8_e4m3* __restrict__ output,
                                    float* __restrict__ scales, uint64_t dim, uint64_t groups,
                                    uint32_t group_size) {
  const uint64_t begin = static_cast<uint64_t>(blockIdx.x) * dim +
                         static_cast<uint64_t>(blockIdx.y) * group_size;
  const uint64_t count =
      min(static_cast<uint64_t>(group_size), dim - static_cast<uint64_t>(blockIdx.y) * group_size);
  float local_amax = 0.0F;
  for (uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
    local_amax = fmaxf(local_amax, fabsf(input[begin + index]));
  }
  __shared__ float shared_amax[kThreads];
  shared_amax[threadIdx.x] = local_amax;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_amax[threadIdx.x] = fmaxf(shared_amax[threadIdx.x], shared_amax[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  __shared__ float shared_scale;
  if (threadIdx.x == 0) {
    shared_scale = shared_amax[0] > 0.0F ? shared_amax[0] / kFp8Max : 1.0F;
    scales[blockIdx.x * groups + blockIdx.y] = shared_scale;
  }
  __syncthreads();
  const float scale = shared_scale;
  for (uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
    const float value = input[begin + index];
    output[begin + index] = __nv_fp8_e4m3(fminf(fmaxf(value / scale, -kFp8Max), kFp8Max));
  }
}

}  // namespace

cudaError_t LaunchFp8Quant(const void* input, void* output, float* scales, uint64_t tokens,
                           uint64_t dim, uint32_t granularity, uint32_t group_size,
                           cudaStream_t stream) {
  if (tokens == 0 || dim == 0) return cudaSuccess;
  if (granularity == 0) {
    Fp8QuantTensorKernel<<<1, kThreads, 0, stream>>>(
        static_cast<const float*>(input), static_cast<__nv_fp8_e4m3*>(output), scales,
        tokens * dim);
    return cudaGetLastError();
  }
  if (granularity == 1) {
    Fp8QuantTokenKernel<<<static_cast<uint32_t>(tokens), kThreads, 0, stream>>>(
        static_cast<const float*>(input), static_cast<__nv_fp8_e4m3*>(output), scales, dim);
    return cudaGetLastError();
  }
  const uint64_t groups = (dim + group_size - 1) / group_size;
  dim3 grid(static_cast<uint32_t>(tokens), static_cast<uint32_t>(groups));
  Fp8QuantGroupKernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const float*>(input), static_cast<__nv_fp8_e4m3*>(output), scales, dim, groups,
      group_size);
  return cudaGetLastError();
}

}  // namespace inferx::kernels::cuda::quant
