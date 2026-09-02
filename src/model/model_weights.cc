#include "inferx/model/model_weights.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::model {
namespace {

absl::Status WeightError(std::string_view field, std::string_view detail) {
  return absl::FailedPreconditionError(absl::StrCat("model_weights.", field, ": ", detail));
}

absl::Status CheckWeight(const TensorView& view, Dtype dtype, std::string_view field, uint64_t rows,
                         uint64_t cols) {
  if (view.dtype() != dtype) {
    return WeightError(field, "dtype does not match the execution family");
  }
  if (view.shape().rank() != 2 || view.shape().dim(0) != rows || view.shape().dim(1) != cols) {
    return WeightError(field, absl::StrCat("expected shape [", rows, ",", cols, "]"));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<ModelWeights> ModelWeights::Create(const LlamaSpec& spec, TensorView embedding,
                                                  TensorView final_norm, TensorView lm_head,
                                                  std::vector<LayerWeights> layers) {
  if (layers.size() != spec.num_hidden_layers) {
    return WeightError("layers", "layer table size mismatch");
  }
  const Dtype dtype = embedding.dtype();
  absl::Status status =
      CheckWeight(embedding, dtype, "embedding", spec.vocab_size, spec.hidden_size);
  if (!status.ok()) return status;
  status = CheckWeight(lm_head, dtype, "lm_head", spec.vocab_size, spec.hidden_size);
  if (!status.ok()) return status;
  if (final_norm.dtype() != dtype || final_norm.shape().rank() != 1 ||
      final_norm.shape().dim(0) != spec.hidden_size) {
    return WeightError("final_norm", "expected shape [hidden]");
  }

  const uint64_t qkv_rows = spec.num_key_value_heads * spec.head_dim;
  const uint64_t query_rows = spec.num_attention_heads * spec.head_dim;
  for (uint32_t layer = 0; layer < spec.num_hidden_layers; ++layer) {
    const std::string prefix = absl::StrCat("layer.", layer, ".");
    const LayerWeights& weight = layers[layer];

    auto check_vector = [&](std::string_view field, const TensorView& view) {
      if (view.dtype() != dtype || view.shape().rank() != 1 ||
          view.shape().dim(0) != spec.hidden_size) {
        return WeightError(absl::StrCat(prefix, field), "expected shape [hidden]");
      }
      return absl::OkStatus();
    };
    // Dense projections are [out, in] over their own role geometry; only
    // the two MLP input projections share the hidden width as the input.
    auto check_matrix = [&](std::string_view field, const TensorView& view, uint64_t rows,
                            uint64_t cols) {
      return CheckWeight(view, dtype, absl::StrCat(prefix, field), rows, cols);
    };

    status = check_vector("input_norm", weight.input_norm);
    if (!status.ok()) return status;
    status = check_vector("post_norm", weight.post_norm);
    if (!status.ok()) return status;
    status = check_matrix("query", weight.query, query_rows, spec.hidden_size);
    if (!status.ok()) return status;
    status = check_matrix("key", weight.key, qkv_rows, spec.hidden_size);
    if (!status.ok()) return status;
    status = check_matrix("value", weight.value, qkv_rows, spec.hidden_size);
    if (!status.ok()) return status;
    status =
        check_matrix("attention_output", weight.attention_output, spec.hidden_size, query_rows);
    if (!status.ok()) return status;
    status = check_matrix("mlp_gate", weight.mlp_gate, spec.intermediate_size, spec.hidden_size);
    if (!status.ok()) return status;
    status = check_matrix("mlp_up", weight.mlp_up, spec.intermediate_size, spec.hidden_size);
    if (!status.ok()) return status;
    status = check_matrix("mlp_down", weight.mlp_down, spec.hidden_size, spec.intermediate_size);
    if (!status.ok()) return status;
  }

  return ModelWeights(embedding, final_norm, lm_head, std::move(layers), dtype);
}

}  // namespace inferx::model
