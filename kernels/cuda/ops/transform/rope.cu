#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "inferx/kernels/cuda/kernel_utils.cuh"
#include "inferx/kernels/cuda/transform_kernels.h"

namespace inferx::kernels::cuda::transform {
namespace {

template <typename T>
__global__ void RopeKernel(const T* input, const int32_t* positions, T* output, uint64_t tokens,
                           uint64_t heads, uint64_t head_dimension, float theta,
                           uint64_t max_position_embeddings) {
  const uint64_t half = head_dimension / 2;
  const uint64_t pair_count = tokens * heads * half;
  const uint64_t pair = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pair_count) return;
  for (uint64_t index = 0; index < tokens; ++index) {
    if (positions[index] < 0 ||
        static_cast<uint64_t>(positions[index]) >= max_position_embeddings) {
      return;
    }
  }
  const uint64_t pair_index = pair % half;
  const uint64_t token_head = pair / half;
  const uint64_t token = token_head / heads;
  const uint64_t base = token_head * head_dimension;
  const float exponent =
      -2.0F * static_cast<float>(pair_index) / static_cast<float>(head_dimension);
  const float angle = static_cast<float>(positions[token]) * powf(theta, exponent);
  float sine = 0.0F;
  float cosine = 0.0F;
  sincosf(angle, &sine, &cosine);
  const float first = Load(input, base + pair_index);
  const float second = Load(input, base + pair_index + half);
  Store(output, base + pair_index, first * cosine - second * sine);
  Store(output, base + pair_index + half, second * cosine + first * sine);
}

template <typename T>
void LaunchRopeTyped(uint32_t query_blocks, uint32_t key_blocks, const void* query, const void* key,
                     const int32_t* positions, void* query_output, void* key_output,
                     uint64_t tokens, uint64_t query_heads, uint64_t kv_heads,
                     uint64_t head_dimension, float theta, uint64_t max_position_embeddings,
                     cudaStream_t stream) {
  RopeKernel<T><<<query_blocks, kKernelThreads, 0, stream>>>(
      static_cast<const T*>(query), positions, static_cast<T*>(query_output), tokens, query_heads,
      head_dimension, theta, max_position_embeddings);
  RopeKernel<T><<<key_blocks, kKernelThreads, 0, stream>>>(
      static_cast<const T*>(key), positions, static_cast<T*>(key_output), tokens, kv_heads,
      head_dimension, theta, max_position_embeddings);
}

}  // namespace

cudaError_t LaunchRopeKernel(const void* query, const void* key, const int32_t* positions,
                             void* query_output, void* key_output, uint64_t tokens,
                             uint64_t query_heads, uint64_t kv_heads, uint64_t head_dimension,
                             float theta, uint64_t max_position_embeddings, StorageType dtype,
                             cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  uint64_t query_work = 0;
  uint64_t key_work = 0;
  cudaError_t status = CheckedWork(tokens, query_heads, &query_work);
  if (status == cudaSuccess) status = CheckedWork(query_work, head_dimension / 2, &query_work);
  if (status == cudaSuccess) status = CheckedWork(tokens, kv_heads, &key_work);
  if (status == cudaSuccess) status = CheckedWork(key_work, head_dimension / 2, &key_work);
  uint32_t query_blocks = 0;
  uint32_t key_blocks = 0;
  if (status == cudaSuccess) status = CheckedBlocks(query_work, &query_blocks);
  if (status == cudaSuccess) status = CheckedBlocks(key_work, &key_blocks);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      LaunchRopeTyped<float>(query_blocks, key_blocks, query, key, positions, query_output,
                             key_output, tokens, query_heads, kv_heads, head_dimension, theta,
                             max_position_embeddings, stream);
      break;
    case StorageType::kFloat16:
      LaunchRopeTyped<__half>(query_blocks, key_blocks, query, key, positions, query_output,
                              key_output, tokens, query_heads, kv_heads, head_dimension, theta,
                              max_position_embeddings, stream);
      break;
    case StorageType::kBFloat16:
      LaunchRopeTyped<__nv_bfloat16>(query_blocks, key_blocks, query, key, positions, query_output,
                                     key_output, tokens, query_heads, kv_heads, head_dimension,
                                     theta, max_position_embeddings, stream);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

}  // namespace inferx::kernels::cuda::transform
