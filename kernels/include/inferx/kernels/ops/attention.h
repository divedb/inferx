// Attention category: prefill/decode attention over a contiguous KV cache.
#ifndef INFERX_KERNELS_OPS_ATTENTION_H_
#define INFERX_KERNELS_OPS_ATTENTION_H_

#include <optional>

#include "absl/status/status.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/provider.h"
#include "inferx/ops/attention.h"

namespace inferx::kernels {

// `metadata` carries device addresses of the indptr/positions/kv-lengths
// arrays; they must stay alive through the caller's completion fence.
[[nodiscard]] absl::Status LaunchAttention(const ops::AttentionRequest& request,
                                           const DeviceAttentionMetadata& metadata,
                                           const KernelExecutionContext& context,
                                           std::optional<ProviderId> forced = std::nullopt);

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_OPS_ATTENTION_H_
