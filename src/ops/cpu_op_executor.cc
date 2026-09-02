#include "inferx/ops/cpu_op_executor.h"

#include <algorithm>
#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::ops {

std::span<float> CpuOpExecutor::AttentionScratch(const AttentionRequest& request) {
  const uint64_t required = AttentionReferenceScratchElements(request);
  if (attention_scratch_.size() < required) {
    attention_scratch_.assign(static_cast<size_t>(required), 0.0F);
  }
  return std::span<float>(attention_scratch_.data(), attention_scratch_.size());
}

absl::Status CpuOpExecutor::Attention(const AttentionRequest& request) {
  return reference_.Execute(request, AttentionScratch(request));
}

absl::Status CpuOpExecutor::Copy(const TensorView& source, const MutableTensorView& destination) {
  return CopyTensorCpu(source, destination);
}

}  // namespace inferx::ops
