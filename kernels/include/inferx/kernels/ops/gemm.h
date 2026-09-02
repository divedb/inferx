// GEMM category: dense matrix products (projectors) and logits projection.
#ifndef INFERX_KERNELS_OPS_GEMM_H_
#define INFERX_KERNELS_OPS_GEMM_H_

#include <optional>

#include "absl/status/status.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/provider.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/logits.h"

namespace inferx::kernels {

[[nodiscard]] absl::Status LaunchGemm(const ops::GemmRequest& request,
                                      const KernelExecutionContext& context,
                                      std::optional<ProviderId> forced = std::nullopt);

[[nodiscard]] absl::Status LaunchLogits(const ops::LogitsRequest& request,
                                        const KernelExecutionContext& context,
                                        std::optional<ProviderId> forced = std::nullopt);

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_OPS_GEMM_H_
