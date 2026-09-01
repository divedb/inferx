// Activation category: elementwise and fused activation kernels.
#ifndef INFERX_KERNELS_OPS_ACTIVATION_H_
#define INFERX_KERNELS_OPS_ACTIVATION_H_

#include <optional>

#include "absl/status/status.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/provider.h"
#include "inferx/ops/swiglu.h"

namespace inferx::kernels {

[[nodiscard]] absl::Status LaunchSwiGlu(const ops::SwiGluRequest& request,
                                        const KernelExecutionContext& context,
                                        std::optional<ProviderId> forced = std::nullopt);

[[nodiscard]] absl::Status LaunchResidual(const ops::ResidualRequest& request,
                                          const KernelExecutionContext& context,
                                          std::optional<ProviderId> forced = std::nullopt);

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_OPS_ACTIVATION_H_
