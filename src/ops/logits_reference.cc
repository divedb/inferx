#include <optional>

#include "absl/status/status.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/logits.h"

namespace inferx::ops {

absl::Status ValidateLogits(const LogitsRequest& request) {
  return ValidateGemm(
      GemmRequest{request.hidden, request.weight, std::nullopt, request.output, 1.0F, 0.0F});
}

absl::Status ReferenceLogits(const LogitsRequest& request) {
  absl::Status status = ValidateLogits(request);
  if (!status.ok()) return status;
  return ReferenceGemm(
      GemmRequest{request.hidden, request.weight, std::nullopt, request.output, 1.0F, 0.0F});
}

}  // namespace inferx::ops
