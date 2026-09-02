#ifndef INFERX_OPS_ATTENTION_H_
#define INFERX_OPS_ATTENTION_H_

#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "inferx/ops/op_context.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::ops {

struct AttentionRequest {
  TensorView query;
  TensorView new_key;
  TensorView new_value;
  std::span<const int32_t> query_indptr;
  std::span<const int32_t> new_kv_indptr;
  std::span<const int32_t> query_positions;
  std::span<const int32_t> kv_lengths_before;
  MutableTensorView key_cache;
  MutableTensorView value_cache;
  MutableTensorView output;
  ExecutionPhase phase = ExecutionPhase::kPrefill;
};

[[nodiscard]] absl::Status ValidateAttention(const AttentionRequest& request);
[[nodiscard]] absl::Status ReferenceAttention(const AttentionRequest& request,
                                              std::span<float> scratch);
[[nodiscard]] uint64_t AttentionReferenceScratchElements(const AttentionRequest& request) noexcept;

}  // namespace inferx::ops

#endif  // INFERX_OPS_ATTENTION_H_
