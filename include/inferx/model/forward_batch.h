#pragma once

#include <cstdint>
#include <span>

#include "inferx/model/intermediate_trace.h"
#include "inferx/scheduler/work_kind.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::model {

// Named activation buffers for one forward step, sized for the step's token
// count T. All buffers are contiguous row-major FP32 in the first M5
// execution family (m5.md section 5); the semantic path never owns them.
struct ActivationBuffers {
  MutableTensorView hidden;      // [T, H]
  MutableTensorView normed;      // [T, H]
  MutableTensorView query;       // [T, Nq, D]
  MutableTensorView key;         // [T, Nkv, D]
  MutableTensorView value;       // [T, Nkv, D]
  MutableTensorView attn_out;    // [T, Nq, D]
  MutableTensorView projected;   // [T, H]
  MutableTensorView gate;        // [T, I]
  MutableTensorView up;          // [T, I]
  MutableTensorView activated;   // [T, I]
  MutableTensorView down_out;    // [T, H]
  MutableTensorView final_norm;  // [T, H]
  MutableTensorView logits;      // [V] FP32 final-row logits
};

// Batch-one forward input: validated token ids and positions (device tensor
// plus host mirror), per-layer contiguous KV cache views, and the append
// range [kv_append_begin, kv_append_begin + T).
struct ForwardBatch {
  WorkKind work = WorkKind::kPrefill;
  uint64_t num_tokens = 0;
  TensorView token_ids;  // [T] int32, execution-device resident
  TensorView positions;  // [T] int32, execution-device resident
  std::span<const int32_t> host_token_ids;
  std::span<const int32_t> host_positions;
  std::span<MutableTensorView> key_cache;    // layers x [1, C, Nkv, D]
  std::span<MutableTensorView> value_cache;  // layers x [1, C, Nkv, D]
  uint64_t kv_append_begin = 0;
  const ActivationBuffers* activations = nullptr;
  IntermediateTraceSink* trace = nullptr;  // optional; null in production
};

}  // namespace inferx::model
