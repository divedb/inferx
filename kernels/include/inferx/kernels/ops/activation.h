// Activation category contracts for kernels-layer operators.
// Fused-layout activation-and-mul matches the [tokens, 2*d] convention used
// by FlashInfer/TokenSpeed; the separate-tensor SwiGlu stays in ops/.
#ifndef INFERX_KERNELS_OPS_ACTIVATION_CONTRACTS_H_
#define INFERX_KERNELS_OPS_ACTIVATION_CONTRACTS_H_

#include <optional>

#include "absl/status/status.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/provider.h"
#include "inferx/ops/swiglu.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::kernels {

enum class ActMulKind : uint8_t { kSilu, kGelu, kGeluTanh };

// input [tokens, 2*d] (gate = cols [0,d), up = cols [d, 2d)) -> [tokens, d].
struct ActMulRequest {
  TensorView input;
  MutableTensorView output;
  ActMulKind kind = ActMulKind::kSilu;
};

struct Add3Request {
  TensorView a;
  TensorView b;
  TensorView c;
  MutableTensorView output;
};

[[nodiscard]] absl::Status LaunchSwiGlu(const ops::SwiGluRequest& request,
                                        const KernelExecutionContext& context,
                                        std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchResidual(const ops::ResidualRequest& request,
                                          const KernelExecutionContext& context,
                                          std::optional<ProviderId> forced = std::nullopt);

[[nodiscard]] absl::Status ValidateActMul(const ActMulRequest& request);
[[nodiscard]] absl::Status ReferenceActMul(const ActMulRequest& request);
[[nodiscard]] absl::Status ValidateAdd3(const Add3Request& request);
[[nodiscard]] absl::Status ReferenceAdd3(const Add3Request& request);

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_OPS_ACTIVATION_CONTRACTS_H_
