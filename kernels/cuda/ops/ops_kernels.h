#ifndef INFERX_KERNELS_CUDA_OPS_OPS_KERNELS_H_
#define INFERX_KERNELS_CUDA_OPS_OPS_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

namespace inferx::cuda::ops::kernels {

enum class StorageType : uint8_t { kFloat32, kFloat16, kBFloat16 };

cudaError_t LaunchEmbeddingKernel(const int32_t* ids, const void* weight, void* output,
                                  uint64_t tokens, uint64_t vocabulary, uint64_t hidden,
                                  StorageType dtype, cudaStream_t stream);
cudaError_t LaunchRmsNormKernel(const void* input, const void* weight, void* output,
                                uint64_t tokens, uint64_t hidden, float epsilon, StorageType dtype,
                                cudaStream_t stream);
cudaError_t LaunchRopeKernel(const void* query, const void* key, const int32_t* positions,
                             void* query_output, void* key_output, uint64_t tokens,
                             uint64_t query_heads, uint64_t kv_heads, uint64_t head_dimension,
                             float theta, uint64_t max_position_embeddings, StorageType dtype,
                             cudaStream_t stream);
cudaError_t LaunchSwiGluKernel(const void* gate, const void* up, void* output, uint64_t elements,
                               StorageType dtype, cudaStream_t stream);
cudaError_t LaunchResidualKernel(const void* left, const void* right, void* output,
                                 uint64_t elements, StorageType dtype, cudaStream_t stream);
cudaError_t LaunchKvAppendKernel(const void* new_key, const void* new_value, void* key_cache,
                                 void* value_cache, const int32_t* new_kv_indptr,
                                 const int32_t* kv_lengths_before, uint64_t batch,
                                 uint64_t total_new, uint64_t max_context, uint64_t kv_heads,
                                 uint64_t head_dimension, StorageType dtype, cudaStream_t stream);
cudaError_t LaunchAttentionKernel(const void* query, const void* key_cache, const void* value_cache,
                                  void* output, const int32_t* query_indptr,
                                  const int32_t* query_positions, uint64_t batch,
                                  uint64_t total_queries, uint64_t max_context,
                                  uint64_t query_heads, uint64_t kv_heads, uint64_t head_dimension,
                                  StorageType dtype, cudaStream_t stream);

}  // namespace inferx::cuda::ops::kernels

#endif  // INFERX_KERNELS_CUDA_OPS_OPS_KERNELS_H_
