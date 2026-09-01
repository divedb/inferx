#ifndef INFERX_OPS_LOGITS_H_
#define INFERX_OPS_LOGITS_H_

#include "absl/status/status.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::ops {

struct LogitsRequest {
  TensorView hidden;
  TensorView weight;
  MutableTensorView output;
};

[[nodiscard]] absl::Status ValidateLogits(const LogitsRequest& request);
[[nodiscard]] absl::Status ReferenceLogits(const LogitsRequest& request);

}  // namespace inferx::ops

#endif  // INFERX_OPS_LOGITS_H_
