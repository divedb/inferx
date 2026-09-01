#ifndef INFERX_OPS_CPU_REFERENCE_EXECUTOR_H_
#define INFERX_OPS_CPU_REFERENCE_EXECUTOR_H_

#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/embedding.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/logits.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/ops/rope.h"
#include "inferx/ops/swiglu.h"

namespace inferx::ops {

struct ReferenceLimits {
  uint64_t maximum_elements = 64ULL * 1024 * 1024;
  uint64_t maximum_flops = 16ULL * 1024 * 1024 * 1024;
};

class CpuReferenceExecutor {
 public:
  explicit CpuReferenceExecutor(ReferenceLimits limits = {}) noexcept : limits_(limits) {}

  [[nodiscard]] absl::Status Execute(const EmbeddingRequest& request) const;
  [[nodiscard]] absl::Status Execute(const GemmRequest& request) const;
  [[nodiscard]] absl::Status Execute(const RmsNormRequest& request) const;
  [[nodiscard]] absl::Status Execute(const RopeRequest& request) const;
  [[nodiscard]] absl::Status Execute(const SiluRequest& request) const;
  [[nodiscard]] absl::Status ExecuteMultiply(const MultiplyRequest& request) const;
  [[nodiscard]] absl::Status ExecuteSwiGlu(const SwiGluRequest& request) const;
  [[nodiscard]] absl::Status ExecuteResidual(const ResidualRequest& request) const;
  [[nodiscard]] absl::Status Execute(const AttentionRequest& request,
                                     std::span<float> scratch) const;
  [[nodiscard]] absl::Status Execute(const LogitsRequest& request) const;

 private:
  [[nodiscard]] absl::Status CheckElements(uint64_t elements) const;
  [[nodiscard]] absl::Status CheckFlops(uint64_t flops) const;
  ReferenceLimits limits_;
};

}  // namespace inferx::ops

#endif  // INFERX_OPS_CPU_REFERENCE_EXECUTOR_H_
