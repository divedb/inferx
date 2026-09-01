#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "ops_kernels.h"

namespace inferx::cuda::ops::kernels {
namespace {

template <typename T>
__device__ float Load(const T* values, uint64_t index) {
  return static_cast<float>(values[index]);
}

template <>
__device__ float Load<__half>(const __half* values, uint64_t index) {
  return __half2float(values[index]);
}

template <>
__device__ float Load<__nv_bfloat16>(const __nv_bfloat16* values, uint64_t index) {
  return __bfloat162float(values[index]);
}

template <typename T>
__device__ void Store(T* values, uint64_t index, float value) {
  values[index] = static_cast<T>(value);
}

template <>
__device__ void Store<__half>(__half* values, uint64_t index, float value) {
  values[index] = __float2half_rn(value);
}

template <>
__device__ void Store<__nv_bfloat16>(__nv_bfloat16* values, uint64_t index, float value) {
  values[index] = __float2bfloat16_rn(value);
}

template <typename T>
__global__ void EmbeddingKernel(const int32_t* ids, const T* weight, T* output, uint64_t tokens,
                                uint64_t vocabulary, uint64_t hidden) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t elements = tokens * hidden;
  if (index >= elements) return;
  // This owned correctness kernel favors transactional semantics over speed:
  // every output thread proves the immutable ID vector valid before any write.
  for (uint64_t candidate = 0; candidate < tokens; ++candidate) {
    if (ids[candidate] < 0 || static_cast<uint64_t>(ids[candidate]) >= vocabulary) return;
  }
  const uint64_t token = index / hidden;
  const uint64_t column = index % hidden;
  const int32_t id = ids[token];
  if (id >= 0 && static_cast<uint64_t>(id) < vocabulary) {
    output[index] = weight[static_cast<uint64_t>(id) * hidden + column];
  }
}

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
__global__ void SwiGluKernel(const T* gate, const T* up, T* output, uint64_t elements) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float value = Load(gate, index);
  Store(output, index, (value / (1.0F + expf(-value))) * Load(up, index));
}

template <typename T>
__global__ void ResidualKernel(const T* left, const T* right, T* output, uint64_t elements) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) Store(output, index, Load(left, index) + Load(right, index));
}

template <typename T>
__global__ void KvAppendKernel(const T* new_key, const T* new_value, T* key_cache, T* value_cache,
                               const int32_t* new_kv_indptr, const int32_t* kv_lengths_before,
                               uint64_t batch, uint64_t total_new, uint64_t max_context,
                               uint64_t kv_heads, uint64_t head_dimension) {
  const uint64_t width = kv_heads * head_dimension;
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_new * width) return;
  const uint64_t token = index / width;
  const uint64_t inner = index % width;
  uint64_t sequence = 0;
  while (sequence + 1 < batch && token >= static_cast<uint64_t>(new_kv_indptr[sequence + 1])) {
    ++sequence;
  }
  const uint64_t local = token - static_cast<uint64_t>(new_kv_indptr[sequence]);
  const uint64_t position = static_cast<uint64_t>(kv_lengths_before[sequence]) + local;
  const uint64_t destination = (sequence * max_context + position) * width + inner;
  key_cache[destination] = new_key[index];
  value_cache[destination] = new_value[index];
}

template <typename T>
__global__ void AttentionKernel(const T* query, const T* key_cache, const T* value_cache, T* output,
                                const int32_t* query_indptr, const int32_t* query_positions,
                                uint64_t batch, uint64_t total_queries, uint64_t max_context,
                                uint64_t query_heads, uint64_t kv_heads, uint64_t head_dimension) {
  const uint64_t work = total_queries * query_heads;
  const uint64_t item = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= work) return;
  const uint64_t query_index = item / query_heads;
  const uint64_t query_head = item % query_heads;
  uint64_t sequence = 0;
  while (sequence + 1 < batch && query_index >= static_cast<uint64_t>(query_indptr[sequence + 1])) {
    ++sequence;
  }
  const uint64_t position = static_cast<uint64_t>(query_positions[query_index]);
  const uint64_t kv_head = query_head / (query_heads / kv_heads);
  const float scale = rsqrtf(static_cast<float>(head_dimension));
  float maximum = -__int_as_float(0x7f800000);
  bool has_nan = false;
  for (uint64_t key_index = 0; key_index <= position; ++key_index) {
    float score = 0.0F;
    for (uint64_t dimension = 0; dimension < head_dimension; ++dimension) {
      const uint64_t query_offset =
          (query_index * query_heads + query_head) * head_dimension + dimension;
      const uint64_t key_offset =
          (((sequence * max_context + key_index) * kv_heads + kv_head) * head_dimension) +
          dimension;
      score += Load(query, query_offset) * Load(key_cache, key_offset);
    }
    const float scaled_score = score * scale;
    has_nan = has_nan || isnan(scaled_score);
    maximum = fmaxf(maximum, scaled_score);
  }
  if (has_nan) maximum = __int_as_float(0x7fc00000);
  float denominator = 0.0F;
  for (uint64_t key_index = 0; key_index <= position; ++key_index) {
    float score = 0.0F;
    for (uint64_t dimension = 0; dimension < head_dimension; ++dimension) {
      const uint64_t query_offset =
          (query_index * query_heads + query_head) * head_dimension + dimension;
      const uint64_t key_offset =
          (((sequence * max_context + key_index) * kv_heads + kv_head) * head_dimension) +
          dimension;
      score += Load(query, query_offset) * Load(key_cache, key_offset);
    }
    denominator += expf(score * scale - maximum);
  }
  for (uint64_t dimension = 0; dimension < head_dimension; ++dimension) {
    float result = 0.0F;
    for (uint64_t key_index = 0; key_index <= position; ++key_index) {
      float score = 0.0F;
      for (uint64_t inner = 0; inner < head_dimension; ++inner) {
        const uint64_t query_offset =
            (query_index * query_heads + query_head) * head_dimension + inner;
        const uint64_t key_offset =
            (((sequence * max_context + key_index) * kv_heads + kv_head) * head_dimension) + inner;
        score += Load(query, query_offset) * Load(key_cache, key_offset);
      }
      const uint64_t value_offset =
          (((sequence * max_context + key_index) * kv_heads + kv_head) * head_dimension) +
          dimension;
      result += (expf(score * scale - maximum) / denominator) * Load(value_cache, value_offset);
    }
    Store(output, (query_index * query_heads + query_head) * head_dimension + dimension, result);
  }
}

constexpr uint32_t kThreads = 128;
constexpr uint64_t kMaximumGridX = 2'147'483'647ULL;

cudaError_t CheckedWork(uint64_t left, uint64_t right, uint64_t* output) {
  if (left != 0 && right > UINT64_MAX / left) return cudaErrorInvalidValue;
  *output = left * right;
  return cudaSuccess;
}

cudaError_t CheckedBlocks(uint64_t work, uint32_t* blocks) {
  if (work == 0) {
    *blocks = 0;
    return cudaSuccess;
  }
  const uint64_t count = (work - 1) / kThreads + 1;
  if (count > kMaximumGridX) return cudaErrorInvalidConfiguration;
  *blocks = static_cast<uint32_t>(count);
  return cudaSuccess;
}

template <typename T>
void LaunchEmbeddingTyped(uint32_t blocks, const int32_t* ids, const void* weight, void* output,
                          uint64_t tokens, uint64_t vocabulary, uint64_t hidden,
                          cudaStream_t stream) {
  EmbeddingKernel<<<blocks, kThreads, 0, stream>>>(
      ids, static_cast<const T*>(weight), static_cast<T*>(output), tokens, vocabulary, hidden);
}

template <typename T>
void LaunchRmsNormTyped(uint32_t blocks, const void* input, const void* weight, void* output,
                        uint64_t tokens, uint64_t hidden, float epsilon, cudaStream_t stream) {
  RmsNormKernel<<<blocks, kThreads, 0, stream>>>(static_cast<const T*>(input),
                                                 static_cast<const T*>(weight),
                                                 static_cast<T*>(output), tokens, hidden, epsilon);
}

template <typename T>
void LaunchRopeTyped(uint32_t query_blocks, uint32_t key_blocks, const void* query, const void* key,
                     const int32_t* positions, void* query_output, void* key_output,
                     uint64_t tokens, uint64_t query_heads, uint64_t kv_heads,
                     uint64_t head_dimension, float theta, uint64_t max_position_embeddings,
                     cudaStream_t stream) {
  RopeKernel<<<query_blocks, kThreads, 0, stream>>>(
      static_cast<const T*>(query), positions, static_cast<T*>(query_output), tokens, query_heads,
      head_dimension, theta, max_position_embeddings);
  RopeKernel<<<key_blocks, kThreads, 0, stream>>>(static_cast<const T*>(key), positions,
                                                  static_cast<T*>(key_output), tokens, kv_heads,
                                                  head_dimension, theta, max_position_embeddings);
}

template <typename T>
void LaunchSwiGluTyped(uint32_t blocks, const void* gate, const void* up, void* output,
                       uint64_t elements, cudaStream_t stream) {
  SwiGluKernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const T*>(gate), static_cast<const T*>(up), static_cast<T*>(output), elements);
}

template <typename T>
void LaunchResidualTyped(uint32_t blocks, const void* left, const void* right, void* output,
                         uint64_t elements, cudaStream_t stream) {
  ResidualKernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const T*>(left), static_cast<const T*>(right), static_cast<T*>(output), elements);
}

template <typename T>
void LaunchKvAppendTyped(uint32_t blocks, const void* new_key, const void* new_value,
                         void* key_cache, void* value_cache, const int32_t* new_kv_indptr,
                         const int32_t* kv_lengths_before, uint64_t batch, uint64_t total_new,
                         uint64_t max_context, uint64_t kv_heads, uint64_t head_dimension,
                         cudaStream_t stream) {
  KvAppendKernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const T*>(new_key), static_cast<const T*>(new_value), static_cast<T*>(key_cache),
      static_cast<T*>(value_cache), new_kv_indptr, kv_lengths_before, batch, total_new, max_context,
      kv_heads, head_dimension);
}

template <typename T>
void LaunchAttentionTyped(uint32_t blocks, const void* query, const void* key_cache,
                          const void* value_cache, void* output, const int32_t* query_indptr,
                          const int32_t* query_positions, uint64_t batch, uint64_t total_queries,
                          uint64_t max_context, uint64_t query_heads, uint64_t kv_heads,
                          uint64_t head_dimension, cudaStream_t stream) {
  AttentionKernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const T*>(query), static_cast<const T*>(key_cache),
      static_cast<const T*>(value_cache), static_cast<T*>(output), query_indptr, query_positions,
      batch, total_queries, max_context, query_heads, kv_heads, head_dimension);
}

}  // namespace

cudaError_t LaunchEmbeddingKernel(const int32_t* ids, const void* weight, void* output,
                                  uint64_t tokens, uint64_t vocabulary, uint64_t hidden,
                                  StorageType dtype, cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  uint64_t work = 0;
  cudaError_t status = CheckedWork(tokens, hidden, &work);
  uint32_t blocks = 0;
  if (status == cudaSuccess) status = CheckedBlocks(work, &blocks);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      LaunchEmbeddingTyped<float>(blocks, ids, weight, output, tokens, vocabulary, hidden, stream);
      break;
    case StorageType::kFloat16:
      LaunchEmbeddingTyped<__half>(blocks, ids, weight, output, tokens, vocabulary, hidden, stream);
      break;
    case StorageType::kBFloat16:
      LaunchEmbeddingTyped<__nv_bfloat16>(blocks, ids, weight, output, tokens, vocabulary, hidden,
                                          stream);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

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

cudaError_t LaunchSwiGluKernel(const void* gate, const void* up, void* output, uint64_t elements,
                               StorageType dtype, cudaStream_t stream) {
  if (elements == 0) return cudaSuccess;
  uint32_t blocks = 0;
  cudaError_t status = CheckedBlocks(elements, &blocks);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      LaunchSwiGluTyped<float>(blocks, gate, up, output, elements, stream);
      break;
    case StorageType::kFloat16:
      LaunchSwiGluTyped<__half>(blocks, gate, up, output, elements, stream);
      break;
    case StorageType::kBFloat16:
      LaunchSwiGluTyped<__nv_bfloat16>(blocks, gate, up, output, elements, stream);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

cudaError_t LaunchResidualKernel(const void* left, const void* right, void* output,
                                 uint64_t elements, StorageType dtype, cudaStream_t stream) {
  if (elements == 0) return cudaSuccess;
  uint32_t blocks = 0;
  cudaError_t status = CheckedBlocks(elements, &blocks);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      LaunchResidualTyped<float>(blocks, left, right, output, elements, stream);
      break;
    case StorageType::kFloat16:
      LaunchResidualTyped<__half>(blocks, left, right, output, elements, stream);
      break;
    case StorageType::kBFloat16:
      LaunchResidualTyped<__nv_bfloat16>(blocks, left, right, output, elements, stream);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

cudaError_t LaunchKvAppendKernel(const void* new_key, const void* new_value, void* key_cache,
                                 void* value_cache, const int32_t* new_kv_indptr,
                                 const int32_t* kv_lengths_before, uint64_t batch,
                                 uint64_t total_new, uint64_t max_context, uint64_t kv_heads,
                                 uint64_t head_dimension, StorageType dtype, cudaStream_t stream) {
  if (total_new == 0) return cudaSuccess;
  uint64_t work = 0;
  cudaError_t status = CheckedWork(total_new, kv_heads, &work);
  if (status == cudaSuccess) status = CheckedWork(work, head_dimension, &work);
  uint32_t blocks = 0;
  if (status == cudaSuccess) status = CheckedBlocks(work, &blocks);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      LaunchKvAppendTyped<float>(blocks, new_key, new_value, key_cache, value_cache, new_kv_indptr,
                                 kv_lengths_before, batch, total_new, max_context, kv_heads,
                                 head_dimension, stream);
      break;
    case StorageType::kFloat16:
      LaunchKvAppendTyped<__half>(blocks, new_key, new_value, key_cache, value_cache, new_kv_indptr,
                                  kv_lengths_before, batch, total_new, max_context, kv_heads,
                                  head_dimension, stream);
      break;
    case StorageType::kBFloat16:
      LaunchKvAppendTyped<__nv_bfloat16>(blocks, new_key, new_value, key_cache, value_cache,
                                         new_kv_indptr, kv_lengths_before, batch, total_new,
                                         max_context, kv_heads, head_dimension, stream);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

cudaError_t LaunchAttentionKernel(const void* query, const void* key_cache, const void* value_cache,
                                  void* output, const int32_t* query_indptr,
                                  const int32_t* query_positions, uint64_t batch,
                                  uint64_t total_queries, uint64_t max_context,
                                  uint64_t query_heads, uint64_t kv_heads, uint64_t head_dimension,
                                  StorageType dtype, cudaStream_t stream) {
  if (total_queries == 0) return cudaSuccess;
  uint64_t work = 0;
  cudaError_t status = CheckedWork(total_queries, query_heads, &work);
  uint32_t blocks = 0;
  if (status == cudaSuccess) status = CheckedBlocks(work, &blocks);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      LaunchAttentionTyped<float>(blocks, query, key_cache, value_cache, output, query_indptr,
                                  query_positions, batch, total_queries, max_context, query_heads,
                                  kv_heads, head_dimension, stream);
      break;
    case StorageType::kFloat16:
      LaunchAttentionTyped<__half>(blocks, query, key_cache, value_cache, output, query_indptr,
                                   query_positions, batch, total_queries, max_context, query_heads,
                                   kv_heads, head_dimension, stream);
      break;
    case StorageType::kBFloat16:
      LaunchAttentionTyped<__nv_bfloat16>(
          blocks, query, key_cache, value_cache, output, query_indptr, query_positions, batch,
          total_queries, max_context, query_heads, kv_heads, head_dimension, stream);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

}  // namespace inferx::cuda::ops::kernels
