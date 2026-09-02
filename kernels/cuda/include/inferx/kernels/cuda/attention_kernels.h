// Attention launch shims: FlashInfer paged decode/prefill mapped from the
// InferX contiguous [batch, max_context, kv_heads, head_dim] cache.
#ifndef INFERX_KERNELS_CUDA_ATTENTION_KERNELS_H_
#define INFERX_KERNELS_CUDA_ATTENTION_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/storage_type.h"

namespace inferx::kernels::cuda::attention {

// Workspace (int32 words) shared by both entry points:
//   [0, B)           kv page indices (iota: page b = cache row b)
//   [B, 2B+1)        kv page indptr (iota: one page per sequence)
//   [2B+1, 3B+1)     last_page_len = kv_lengths_before + new tokens
//   [3B+1, 4B+1)     request_indices (decode: iota; prefill: tile mapping)
//   [4B+1, 5B+1)     tile indices (decode: zeros; prefill: qo tiles)
//   [5B+1, P+6B+7)   prefill-only: q_indptr mirror (B+1) + tile_offsets (B+1)
//                    where P = padded tile count
//   last word        kv_chunk_size scalar
[[nodiscard]] uint64_t AttentionWorkspaceWords(uint64_t batch, uint64_t padded_tiles);

// Decode: one query per sequence. FP16/BF16 only (FlashInfer kernels). query [batch, q_heads,
// head_dim]; key_cache/value_cache [batch, max_context, kv_heads, head_dim].
cudaError_t LaunchFlashInferDecode(const void* query, const void* key_cache,
                                   const void* value_cache, const int32_t* new_kv_indptr,
                                   const int32_t* kv_lengths_before, void* output, void* workspace,
                                   uint64_t workspace_words, uint64_t batch, uint64_t max_context,
                                   uint64_t query_heads, uint64_t kv_heads, uint64_t head_dim,
                                   StorageType dtype, cudaStream_t stream);

// Prefill with host staging: host_q_indptr/host_tile_offsets are host
// arrays of length batch+1; tile counts use a 64-query CTA tile. FP16/BF16
// only (FlashInfer kernels).
cudaError_t LaunchFlashInferPrefillStaged(
    const void* query, const void* key_cache, const void* value_cache, const int32_t* host_q_indptr,
    const int32_t* host_tile_offsets, const int32_t* device_q_indptr, const int32_t* new_kv_indptr,
    const int32_t* kv_lengths_before, void* output, void* workspace, uint64_t workspace_words,
    uint64_t batch, uint64_t total_queries, uint64_t max_context, uint64_t query_heads,
    uint64_t kv_heads, uint64_t head_dim, StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::attention

#endif  // INFERX_KERNELS_CUDA_ATTENTION_KERNELS_H_
