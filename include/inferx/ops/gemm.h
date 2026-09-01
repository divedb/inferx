#ifndef INFERX_OPS_GEMM_H_
#define INFERX_OPS_GEMM_H_

#include <optional>

#include "absl/status/status.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::ops {

struct GemmRequest {
  TensorView input;
  TensorView weight;
  std::optional<TensorView> addend;
  MutableTensorView output;
  float alpha = 1.0F;
  float beta = 0.0F;
};

[[nodiscard]] absl::Status ValidateGemm(const GemmRequest& request);
[[nodiscard]] absl::Status ReferenceGemm(const GemmRequest& request);

}  // namespace inferx::ops

#endif  // INFERX_OPS_GEMM_H_
