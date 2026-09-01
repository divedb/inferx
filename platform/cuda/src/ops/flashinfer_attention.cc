#include "inferx/platform/cuda/ops/flashinfer_attention.h"

#include "absl/status/status.h"

namespace inferx::cuda::ops {

absl::Status FlashInferAvailability() {
  return absl::UnimplementedError(
      "flashinfer: ADR 0029 rejects the recorded pin for this production build");
}

}  // namespace inferx::cuda::ops
