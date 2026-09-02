// Batch decode/prefill attention via FlashInfer's device kernels (ADR 0032
// reuse tier 2), closing the recorded paged-KV gap: the InferX contiguous
// [batch, max_context, kv_heads, head_dim] cache maps exactly onto
// paged_kv_t with page_size = max_context and one page per sequence, and a
// marshalling kernel fills the page tables / scheduler arrays inside the
// caller workspace. Direct (non-planned) launches: no split-KV inside the
// 4096-token envelope.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/attention/default_prefill_params.cuh"
#include "flashinfer/attention/prefill.cuh"
#include "flashinfer/attention/variants.cuh"
#include "inferx/kernels/cuda/attention_kernels.h"

namespace inferx::kernels::cuda::attention {
namespace {

using AttentionVariant = flashinfer::DefaultAttention</*use_custom_mask=*/false,
                                                      /*use_sliding_window=*/false,
                                                      /*use_logits_soft_cap=*/false,
                                                      /*use_alibi=*/false>;

constexpr uint32_t kCtaTileQ = 64;

// Marshalling kernel, one thread per item.
//   index < batch:               page-table entries for request `index`
//   index == batch:              kv_indptr terminator
//   index in (batch, batch+1+tokens]: prefill tile enumeration via
//                                 binary search over device tile offsets
__global__ void AttentionMetadataKernel(int32_t* kv_indices, int32_t* kv_indptr,
                                        int32_t* last_page_len, int32_t* request_indices,
                                        int32_t* qo_tile_indices, int32_t* kv_tile_indices,
                                        int32_t* kv_chunk_size, const int32_t* device_q_indptr,
                                        const int32_t* device_tile_offsets,
                                        const int32_t* new_kv_indptr,
                                        const int32_t* kv_lengths_before, uint32_t batch,
                                        uint32_t tiles) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < batch) {
    kv_indices[index] = static_cast<int32_t>(index);
    kv_indptr[index] = static_cast<int32_t>(index);
    last_page_len[index] =
        kv_lengths_before[index] + (new_kv_indptr[index + 1] - new_kv_indptr[index]);
    return;
  }
  if (index == batch) {
    kv_indptr[batch] = static_cast<int32_t>(batch);
    return;
  }
  const uint32_t tile = index - (batch + 1);
  if (tile >= tiles) return;
  uint32_t lo = 0;
  uint32_t hi = batch;
  while (lo + 1 < hi) {
    const uint32_t mid = (lo + hi) / 2;
    if (static_cast<uint32_t>(device_tile_offsets[mid]) <= tile) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const uint32_t qo_len = device_q_indptr[lo + 1] - device_q_indptr[lo];
  request_indices[tile] = static_cast<int32_t>(lo);
  qo_tile_indices[tile] = static_cast<int32_t>(tile - device_tile_offsets[lo]);
  kv_tile_indices[tile] = 0;
  (void)qo_len;
}

struct Workspace {
  int32_t* kv_indices;       // [batch]
  int32_t* kv_indptr;        // [batch + 1]
  int32_t* last_page_len;    // [batch]
  int32_t* request_indices;  // [tiles]
  int32_t* qo_tile_indices;  // [tiles]
  int32_t* kv_tile_indices;  // [tiles]
  int32_t* prefill_aux;      // q_indptr mirror + tile offsets ([batch+1] each)
  int32_t* kv_chunk_size;    // scalar
};

Workspace SplitWorkspace(void* workspace, uint64_t batch, uint64_t tiles) {
  auto* base = static_cast<int32_t*>(workspace);
  Workspace w{};
  w.kv_indices = base;
  w.kv_indptr = w.kv_indices + batch;
  w.last_page_len = w.kv_indptr + batch + 1;
  w.request_indices = w.last_page_len + batch;
  w.qo_tile_indices = w.request_indices + tiles;
  w.kv_tile_indices = w.qo_tile_indices + tiles;
  w.prefill_aux = w.kv_tile_indices + tiles;
  w.kv_chunk_size = w.prefill_aux + 2 * (batch + 1);
  return w;
}

template <typename DType>
flashinfer::paged_kv_t<DType, int32_t> MakePagedKv(const Workspace& w, const void* key_cache,
                                                   const void* value_cache, uint64_t batch,
                                                   uint64_t max_context, uint64_t kv_heads,
                                                   uint64_t head_dim) {
  return flashinfer::paged_kv_t<DType, int32_t>(
      static_cast<uint32_t>(kv_heads), static_cast<uint32_t>(max_context),
      static_cast<uint32_t>(head_dim), static_cast<uint32_t>(batch), flashinfer::QKVLayout::kNHD,
      static_cast<DType*>(const_cast<void*>(key_cache)),
      static_cast<DType*>(const_cast<void*>(value_cache)), w.kv_indices, w.kv_indptr,
      w.last_page_len);
}

template <typename DType>
cudaError_t DispatchDecode(const Workspace& w, const void* query, const void* key_cache,
                           const void* value_cache, void* output, uint64_t batch,
                           uint64_t max_context, uint64_t query_heads, uint64_t kv_heads,
                           uint64_t head_dim, cudaStream_t stream) {
  auto paged_kv =
      MakePagedKv<DType>(w, key_cache, value_cache, batch, max_context, kv_heads, head_dim);
  flashinfer::BatchDecodeParams<DType, DType, DType, int32_t> params(
      static_cast<DType*>(const_cast<void*>(query)), /*q_rope_offset=*/nullptr, paged_kv,
      static_cast<DType*>(output),
      /*lse=*/nullptr, /*maybe_alibi_slopes=*/nullptr, static_cast<uint32_t>(query_heads),
      /*q_stride_n=*/static_cast<int32_t>(query_heads * head_dim),
      /*q_stride_h=*/static_cast<int32_t>(head_dim),
      /*window_left=*/-1, /*logits_soft_cap=*/0.0F,
      /*sm_scale=*/1.0F / sqrtf(static_cast<float>(head_dim)),
      /*rope_scale=*/1.0F, /*rope_theta=*/1.0e4F);
  params.padded_batch_size = static_cast<uint32_t>(batch);
  params.request_indices = w.request_indices;
  params.kv_tile_indices = w.kv_tile_indices;
  params.kv_chunk_size_ptr = w.kv_chunk_size;
  switch (head_dim) {
    case 64:
      return flashinfer::BatchDecodeWithPagedKVCacheDispatched<
          64, flashinfer::PosEncodingMode::kNone, AttentionVariant>(
          params, /*tmp_v=*/nullptr, /*tmp_s=*/nullptr, /*enable_pdl=*/false, stream);
    case 128:
      return flashinfer::BatchDecodeWithPagedKVCacheDispatched<
          128, flashinfer::PosEncodingMode::kNone, AttentionVariant>(params, nullptr, nullptr,
                                                                     false, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

template <typename DType>
cudaError_t DispatchPrefill(const Workspace& w, const void* query, const void* key_cache,
                            const void* value_cache, void* output, uint64_t batch,
                            uint64_t total_queries, uint64_t max_context, uint64_t query_heads,
                            uint64_t kv_heads, uint64_t head_dim, uint32_t tiles,
                            cudaStream_t stream) {
  auto paged_kv =
      MakePagedKv<DType>(w, key_cache, value_cache, batch, max_context, kv_heads, head_dim);
  using Params = flashinfer::BatchPrefillPagedParams<DType, DType, DType, int32_t>;
  Params params(static_cast<DType*>(const_cast<void*>(query)), paged_kv,
                /*maybe_custom_mask=*/nullptr, w.prefill_aux,
                /*maybe_mask_indptr=*/nullptr, /*maybe_q_rope_offset=*/nullptr,
                static_cast<DType*>(output), /*lse=*/nullptr, /*maybe_alibi_slopes=*/nullptr,
                static_cast<uint32_t>(query_heads),
                /*q_stride_n=*/static_cast<int32_t>(query_heads * head_dim),
                /*q_stride_h=*/static_cast<int32_t>(head_dim),
                /*window_left=*/-1, /*logits_soft_cap=*/0.0F,
                /*sm_scale=*/1.0F / sqrtf(static_cast<float>(head_dim)),
                /*rope_scale=*/1.0F, /*rope_theta=*/1.0e4F);
  params.request_indices = w.request_indices;
  params.qo_tile_indices = w.qo_tile_indices;
  params.kv_tile_indices = w.kv_tile_indices;
  params.kv_chunk_size_ptr = w.kv_chunk_size;
  params.max_total_num_rows = static_cast<uint32_t>(total_queries);
  params.padded_batch_size = tiles;
  constexpr bool kUseFp16QkReduction = false;
  switch (head_dim) {
    case 64:
      return flashinfer::BatchPrefillWithPagedKVCacheDispatched<
          kCtaTileQ, 64, 64, flashinfer::PosEncodingMode::kNone, kUseFp16QkReduction,
          flashinfer::MaskMode::kCausal, AttentionVariant>(params, nullptr, nullptr, false, stream);
    case 128:
      return flashinfer::BatchPrefillWithPagedKVCacheDispatched<
          kCtaTileQ, 128, 128, flashinfer::PosEncodingMode::kNone, kUseFp16QkReduction,
          flashinfer::MaskMode::kCausal, AttentionVariant>(params, nullptr, nullptr, false, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t LaunchMarshalling(const Workspace& w, const int32_t* device_q_indptr,
                              const int32_t* device_tile_offsets, const int32_t* new_kv_indptr,
                              const int32_t* kv_lengths_before, uint64_t batch, uint64_t tiles,
                              cudaStream_t stream) {
  const uint64_t total = batch + 1 + tiles;
  const uint32_t blocks = static_cast<uint32_t>((total + 255) / 256);
  AttentionMetadataKernel<<<blocks, 256, 0, stream>>>(
      w.kv_indices, w.kv_indptr, w.last_page_len, w.request_indices, w.qo_tile_indices,
      w.kv_tile_indices, w.kv_chunk_size, device_q_indptr, device_tile_offsets, new_kv_indptr,
      kv_lengths_before, static_cast<uint32_t>(batch), static_cast<uint32_t>(tiles));
  return cudaGetLastError();
}

}  // namespace

uint64_t AttentionWorkspaceWords(uint64_t batch, uint64_t padded_tiles) {
  return batch + (batch + 1) + batch + 3 * padded_tiles + 2 * (batch + 1) + 1;
}

cudaError_t LaunchFlashInferDecode(const void* query, const void* key_cache,
                                   const void* value_cache, const int32_t* new_kv_indptr,
                                   const int32_t* kv_lengths_before, void* output, void* workspace,
                                   uint64_t workspace_words, uint64_t batch, uint64_t max_context,
                                   uint64_t query_heads, uint64_t kv_heads, uint64_t head_dim,
                                   StorageType dtype, cudaStream_t stream) {
  if (batch == 0) return cudaSuccess;
  if (workspace == nullptr || workspace_words < AttentionWorkspaceWords(batch, batch)) {
    return cudaErrorInvalidValue;
  }
  Workspace w = SplitWorkspace(workspace, batch, batch);
  // q/kv tile metadata degenerates to iota/zeros; device_q_indptr unused.
  cudaError_t status = LaunchMarshalling(w, w.prefill_aux, w.prefill_aux + batch + 1, new_kv_indptr,
                                         kv_lengths_before, batch, batch, stream);
  if (status != cudaSuccess) return status;
  if (dtype == StorageType::kFloat32) return cudaErrorInvalidValue;
  switch (dtype) {
    case StorageType::kFloat16:
      return DispatchDecode<__half>(w, query, key_cache, value_cache, output, batch, max_context,
                                    query_heads, kv_heads, head_dim, stream);
    case StorageType::kBFloat16:
      return DispatchDecode<__nv_bfloat16>(w, query, key_cache, value_cache, output, batch,
                                           max_context, query_heads, kv_heads, head_dim, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

// Prefill entry with host staging: host_q_indptr/host_tile_offsets are host
// arrays of length batch+1; tile counts use kCtaTileQ.
cudaError_t LaunchFlashInferPrefillStaged(
    const void* query, const void* key_cache, const void* value_cache, const int32_t* host_q_indptr,
    const int32_t* host_tile_offsets, const int32_t* device_q_indptr, const int32_t* new_kv_indptr,
    const int32_t* kv_lengths_before, void* output, void* workspace, uint64_t workspace_words,
    uint64_t batch, uint64_t total_queries, uint64_t max_context, uint64_t query_heads,
    uint64_t kv_heads, uint64_t head_dim, StorageType dtype, cudaStream_t stream) {
  if (batch == 0 || total_queries == 0) return cudaSuccess;
  const uint64_t tiles = static_cast<uint64_t>(host_tile_offsets[batch]);
  if (workspace == nullptr || workspace_words < AttentionWorkspaceWords(batch, tiles)) {
    return cudaErrorInvalidValue;
  }
  Workspace w = SplitWorkspace(workspace, batch, tiles);
  cudaMemcpyAsync(w.prefill_aux, host_q_indptr, (batch + 1) * sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(w.prefill_aux + batch + 1, host_tile_offsets, (batch + 1) * sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  status = LaunchMarshalling(w, w.prefill_aux, w.prefill_aux + batch + 1, new_kv_indptr,
                             kv_lengths_before, batch, tiles, stream);
  if (status != cudaSuccess) return status;
  if (dtype == StorageType::kFloat32) return cudaErrorInvalidValue;
  switch (dtype) {
    case StorageType::kFloat16:
      return DispatchPrefill<__half>(w, query, key_cache, value_cache, output, batch, total_queries,
                                     max_context, query_heads, kv_heads, head_dim,
                                     static_cast<uint32_t>(tiles), stream);
    case StorageType::kBFloat16:
      return DispatchPrefill<__nv_bfloat16>(w, query, key_cache, value_cache, output, batch,
                                            total_queries, max_context, query_heads, kv_heads,
                                            head_dim, static_cast<uint32_t>(tiles), stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace inferx::kernels::cuda::attention
