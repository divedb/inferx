#pragma once

#include <span>

#include "absl/status/status.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/embedding.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/logits.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/ops/rope.h"
#include "inferx/ops/swiglu.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::ops {

// The M5 execution seam (ADR 0035): model semantics name operations, an
// OpExecutor performs them on one backend. Implementations are the M4 CPU
// reference and the kernels-dispatch device executor; both satisfy the M4
// operator contracts, including validation before mutation and (for device
// launches) queued-not-completed launch semantics owned by the caller's
// completion fence.
class OpExecutor {
 public:
  virtual ~OpExecutor() = default;

  [[nodiscard]] virtual absl::Status Embedding(const EmbeddingRequest& request) = 0;
  [[nodiscard]] virtual absl::Status Gemm(const GemmRequest& request) = 0;
  [[nodiscard]] virtual absl::Status RmsNorm(const RmsNormRequest& request) = 0;
  [[nodiscard]] virtual absl::Status Rope(const RopeRequest& request) = 0;
  [[nodiscard]] virtual absl::Status SwiGlu(const SwiGluRequest& request) = 0;
  [[nodiscard]] virtual absl::Status Residual(const ResidualRequest& request) = 0;
  [[nodiscard]] virtual absl::Status Attention(const AttentionRequest& request) = 0;
  [[nodiscard]] virtual absl::Status Logits(const LogitsRequest& request) = 0;

  // Contiguous same-shape copy; used by the semantic path for buffer
  // management and by backends for host staging. Overlap is rejected.
  [[nodiscard]] virtual absl::Status Copy(const TensorView& source,
                                          const MutableTensorView& destination) = 0;
};

}  // namespace inferx::ops
