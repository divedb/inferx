#ifndef INFERX_OPS_ROPE_H_
#define INFERX_OPS_ROPE_H_

#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::ops {

struct RopeRequest {
  TensorView query;
  TensorView key;
  TensorView positions;
  MutableTensorView query_output;
  MutableTensorView key_output;
  float theta = 10'000.0F;
  uint64_t max_position_embeddings = 0;
  // CUDA callers retain this immutable mirror for synchronous semantic validation.
  std::span<const int32_t> host_positions{};
};

[[nodiscard]] absl::Status ValidateRope(const RopeRequest& request);
[[nodiscard]] absl::Status ReferenceRope(const RopeRequest& request);

}  // namespace inferx::ops

#endif  // INFERX_OPS_ROPE_H_
