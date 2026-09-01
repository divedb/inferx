// Layernorm category: RMS normalization variants.
#ifndef INFERX_KERNELS_OPS_LAYERNORM_H_
#define INFERX_KERNELS_OPS_LAYERNORM_H_

#include <optional>

#include "absl/status/status.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/provider.h"
#include "inferx/ops/rms_norm.h"

namespace inferx::kernels {

[[nodiscard]] absl::Status LaunchRmsNorm(const ops::RmsNormRequest& request,
                                         const KernelExecutionContext& context,
                                         std::optional<ProviderId> forced = std::nullopt);

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_OPS_LAYERNORM_H_
