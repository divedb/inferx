// Transform category: positional/shape transforms of activations (RoPE).
#ifndef INFERX_KERNELS_OPS_TRANSFORM_H_
#define INFERX_KERNELS_OPS_TRANSFORM_H_

#include <optional>

#include "absl/status/status.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/provider.h"
#include "inferx/ops/rope.h"

namespace inferx::kernels {

[[nodiscard]] absl::Status LaunchRope(const ops::RopeRequest& request,
                                      const KernelExecutionContext& context,
                                      std::optional<ProviderId> forced = std::nullopt);

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_OPS_TRANSFORM_H_
