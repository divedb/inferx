#pragma once

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/cpu_reference_executor.h"
#include "inferx/ops/op_executor.h"

namespace inferx::ops {

// Reference OpExecutor over the M4 CPU kernels. The attention scratch is
// caller-opaque here and grown once to the largest seen context; this path is
// the correctness oracle and test surface (m5.md section 16.1), not the
// production executor.
class CpuOpExecutor : public OpExecutor {
 public:
  explicit CpuOpExecutor(ReferenceLimits limits = {}) noexcept : reference_(limits) {}

  [[nodiscard]] absl::Status Embedding(const EmbeddingRequest& request) override {
    return reference_.Execute(request);
  }
  [[nodiscard]] absl::Status Gemm(const GemmRequest& request) override {
    return reference_.Execute(request);
  }
  [[nodiscard]] absl::Status RmsNorm(const RmsNormRequest& request) override {
    return reference_.Execute(request);
  }
  [[nodiscard]] absl::Status Rope(const RopeRequest& request) override {
    return reference_.Execute(request);
  }
  [[nodiscard]] absl::Status SwiGlu(const SwiGluRequest& request) override {
    return reference_.ExecuteSwiGlu(request);
  }
  [[nodiscard]] absl::Status Residual(const ResidualRequest& request) override {
    return reference_.ExecuteResidual(request);
  }
  [[nodiscard]] absl::Status Attention(const AttentionRequest& request) override;
  [[nodiscard]] absl::Status Logits(const LogitsRequest& request) override {
    return reference_.Execute(request);
  }
  [[nodiscard]] absl::Status Copy(const TensorView& source,
                                  const MutableTensorView& destination) override;

 private:
  [[nodiscard]] std::span<float> AttentionScratch(const AttentionRequest& request);

  CpuReferenceExecutor reference_;
  std::vector<float> attention_scratch_;
};

}  // namespace inferx::ops
