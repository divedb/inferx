#ifndef INFERX_OPS_EMBEDDING_H_
#define INFERX_OPS_EMBEDDING_H_

#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::ops {

struct EmbeddingRequest {
  TensorView token_ids;
  TensorView weight;
  MutableTensorView output;
  // CUDA callers retain this immutable mirror for synchronous semantic validation.
  std::span<const int32_t> host_token_ids{};
};

[[nodiscard]] absl::Status ValidateEmbedding(const EmbeddingRequest& request);
[[nodiscard]] absl::Status ReferenceEmbedding(const EmbeddingRequest& request);

}  // namespace inferx::ops

#endif  // INFERX_OPS_EMBEDDING_H_
