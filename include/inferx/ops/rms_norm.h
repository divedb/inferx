#ifndef INFERX_OPS_RMS_NORM_H_
#define INFERX_OPS_RMS_NORM_H_

#include "absl/status/status.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::ops {

struct RmsNormRequest {
  TensorView input;
  TensorView weight;
  MutableTensorView output;
  float epsilon = 0.0F;
};

[[nodiscard]] absl::Status ValidateRmsNorm(const RmsNormRequest& request);
[[nodiscard]] absl::Status ReferenceRmsNorm(const RmsNormRequest& request);

}  // namespace inferx::ops

#endif  // INFERX_OPS_RMS_NORM_H_
