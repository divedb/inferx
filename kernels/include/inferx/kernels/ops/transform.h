// Transform category contract: length-128 Hadamard (Sylvester) transform.
#ifndef INFERX_KERNELS_OPS_TRANSFORM_CONTRACTS_H_
#define INFERX_KERNELS_OPS_TRANSFORM_CONTRACTS_H_

#include <optional>

#include "absl/status/status.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/provider.h"
#include "inferx/ops/rope.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::kernels {

// x [..., 128] -> same-shape output; out = H @ x * scale where
// H[i][j] = (-1)^popcount(i & j).
struct HadamardTransformRequest {
  TensorView input;
  MutableTensorView output;
  float scale = 1.0F;
};

[[nodiscard]] absl::Status LaunchRope(const ops::RopeRequest& request,
                                      const KernelExecutionContext& context,
                                      std::optional<ProviderId> forced = std::nullopt);

[[nodiscard]] absl::Status ValidateHadamardTransform(const HadamardTransformRequest& request);
[[nodiscard]] absl::Status ReferenceHadamardTransform(const HadamardTransformRequest& request);

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_OPS_TRANSFORM_CONTRACTS_H_
