#ifndef INFERX_PLATFORM_CUDA_OPS_CUDA_ATTENTION_H_
#define INFERX_PLATFORM_CUDA_OPS_CUDA_ATTENTION_H_

#include <span>

#include "absl/status/status.h"
#include "inferx/ops/attention.h"
#include "inferx/platform/cuda/ops/cuda_op_context.h"

namespace inferx::cuda::ops {

// Correctness-oriented contiguous-KV fallback. Metadata spans must refer to
// device memory and remain alive through the caller's completion fence.
struct DeviceAttentionMetadata {
  const int32_t* query_indptr = nullptr;
  const int32_t* new_kv_indptr = nullptr;
  const int32_t* query_positions = nullptr;
  const int32_t* kv_lengths_before = nullptr;
};

[[nodiscard]] absl::Status LaunchAttentionFallback(const inferx::ops::AttentionRequest& request,
                                                   const DeviceAttentionMetadata& metadata,
                                                   const CudaOpContext& context);

}  // namespace inferx::cuda::ops

#endif  // INFERX_PLATFORM_CUDA_OPS_CUDA_ATTENTION_H_
