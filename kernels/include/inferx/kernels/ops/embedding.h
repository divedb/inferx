// Embedding category: token-id gather from a row-major vocabulary table.
#ifndef INFERX_KERNELS_OPS_EMBEDDING_H_
#define INFERX_KERNELS_OPS_EMBEDDING_H_

#include <optional>

#include "absl/status/status.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/provider.h"
#include "inferx/ops/embedding.h"

namespace inferx::kernels {

[[nodiscard]] absl::Status LaunchEmbedding(const ops::EmbeddingRequest& request,
                                           const KernelExecutionContext& context,
                                           std::optional<ProviderId> forced = std::nullopt);

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_OPS_EMBEDDING_H_
