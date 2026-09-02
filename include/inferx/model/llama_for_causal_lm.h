#pragma once

#include "absl/status/status.h"
#include "inferx/model/forward_batch.h"
#include "inferx/model/model_spec.h"
#include "inferx/model/model_weights.h"
#include "inferx/ops/op_executor.h"

namespace inferx::model {

// Immutable dense-Llama semantic composition (m5.md section 8). It owns no
// memory, chooses no kernel backend, allocates no KV, and samples nothing:
// every buffer arrives through the ForwardBatch/ModelWeights views and every
// operation runs through the OpExecutor.
class LlamaForCausalLM {
 public:
  static absl::StatusOr<LlamaForCausalLM> Create(ModelSpec spec);

  // Executes the exact forward order for one batch. RoPE runs before the KV
  // append so cached K is rotated exactly once; the ordinary-generation path
  // computes the final logical row through the LM head only.
  [[nodiscard]] absl::Status Forward(const ForwardBatch& batch, const ModelWeights& weights,
                                     ops::OpExecutor& executor) const;

  [[nodiscard]] const ModelSpec& spec() const noexcept { return spec_; }

 private:
  explicit LlamaForCausalLM(ModelSpec spec) : spec_(std::move(spec)) {}

  ModelSpec spec_;
};

}  // namespace inferx::model
