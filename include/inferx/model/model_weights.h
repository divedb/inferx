#pragma once

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/model/model_spec.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::model {

// Immutable per-layer weight views keyed by semantic role. The table is
// resolved once at model-instance construction from the backend's weight
// arena plus the M3 alias table; the forward path performs no name lookup.
struct LayerWeights {
  TensorView input_norm;
  TensorView query;
  TensorView key;
  TensorView value;
  TensorView attention_output;
  TensorView post_norm;
  TensorView mlp_gate;
  TensorView mlp_up;
  TensorView mlp_down;
};

class ModelWeights {
 public:
  // Validates every view against the spec (shape and dtype uniformity).
  // `lm_head` may be the same view as `embedding` for tied checkpoints.
  static absl::StatusOr<ModelWeights> Create(const LlamaSpec& spec, TensorView embedding,
                                             TensorView final_norm, TensorView lm_head,
                                             std::vector<LayerWeights> layers);

  [[nodiscard]] const TensorView& embedding() const noexcept { return embedding_; }
  [[nodiscard]] const TensorView& final_norm() const noexcept { return final_norm_; }
  [[nodiscard]] const TensorView& lm_head() const noexcept { return lm_head_; }
  [[nodiscard]] const std::vector<LayerWeights>& layers() const noexcept { return layers_; }
  [[nodiscard]] Dtype dtype() const noexcept { return dtype_; }

 private:
  ModelWeights(TensorView embedding, TensorView final_norm, TensorView lm_head,
               std::vector<LayerWeights> layers, Dtype dtype)
      : embedding_(embedding),
        final_norm_(final_norm),
        lm_head_(lm_head),
        layers_(std::move(layers)),
        dtype_(dtype) {}

  TensorView embedding_;
  TensorView final_norm_;
  TensorView lm_head_;
  std::vector<LayerWeights> layers_;
  Dtype dtype_ = Dtype::kFloat32;
};

}  // namespace inferx::model
