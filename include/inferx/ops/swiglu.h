#ifndef INFERX_OPS_SWIGLU_H_
#define INFERX_OPS_SWIGLU_H_

#include "absl/status/status.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::ops {

struct SiluRequest {
  TensorView input;
  MutableTensorView output;
};

struct MultiplyRequest {
  TensorView left;
  TensorView right;
  MutableTensorView output;
};

using SwiGluRequest = MultiplyRequest;
using ResidualRequest = MultiplyRequest;

[[nodiscard]] absl::Status ValidateSilu(const SiluRequest& request);
[[nodiscard]] absl::Status ReferenceSilu(const SiluRequest& request);
[[nodiscard]] absl::Status ValidateMultiply(const MultiplyRequest& request);
[[nodiscard]] absl::Status ReferenceMultiply(const MultiplyRequest& request);
[[nodiscard]] absl::Status ValidateSwiGlu(const SwiGluRequest& request);
[[nodiscard]] absl::Status ReferenceSwiGlu(const SwiGluRequest& request);
[[nodiscard]] absl::Status ValidateResidual(const ResidualRequest& request);
[[nodiscard]] absl::Status ReferenceResidual(const ResidualRequest& request);

}  // namespace inferx::ops

#endif  // INFERX_OPS_SWIGLU_H_
