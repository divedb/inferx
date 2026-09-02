// Layernorm category contracts beyond plain RMSNorm.
#ifndef INFERX_KERNELS_OPS_LAYERNORM_CONTRACTS_H_
#define INFERX_KERNELS_OPS_LAYERNORM_CONTRACTS_H_

#include <optional>

#include "absl/status/status.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/provider.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::kernels {

// residual += input; output = rmsnorm(input + residual) * weight, both
// in-place (FlashInfer fused_add_rmsnorm semantics).
struct FusedAddRmsNormRequest {
  MutableTensorView input;
  MutableTensorView residual;
  TensorView weight;
  float epsilon = 1.0e-5F;
};

// Gemma-style (weight + 1) multiplier RMSNorm.
struct GemmaRmsNormRequest {
  TensorView input;
  TensorView weight;
  MutableTensorView output;
  float epsilon = 1.0e-5F;
};

// Per-head RMSNorm over q [tokens, q_heads, head_dim] and k
// [tokens, kv_heads, head_dim] with separate weights.
struct QkRmsNormRequest {
  MutableTensorView query;
  MutableTensorView key;
  TensorView query_weight;
  TensorView key_weight;
  float epsilon = 1.0e-5F;
};

[[nodiscard]] absl::Status LaunchRmsNorm(const ops::RmsNormRequest& request,
                                         const KernelExecutionContext& context,
                                         std::optional<ProviderId> forced = std::nullopt);

[[nodiscard]] absl::Status ValidateFusedAddRmsNorm(const FusedAddRmsNormRequest& request);
[[nodiscard]] absl::Status ReferenceFusedAddRmsNorm(const FusedAddRmsNormRequest& request);
[[nodiscard]] absl::Status ValidateGemmaRmsNorm(const GemmaRmsNormRequest& request);
[[nodiscard]] absl::Status ReferenceGemmaRmsNorm(const GemmaRmsNormRequest& request);
[[nodiscard]] absl::Status ValidateQkRmsNorm(const QkRmsNormRequest& request);
[[nodiscard]] absl::Status ReferenceQkRmsNorm(const QkRmsNormRequest& request);

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_OPS_LAYERNORM_CONTRACTS_H_
