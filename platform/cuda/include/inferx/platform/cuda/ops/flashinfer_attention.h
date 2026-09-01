#ifndef INFERX_PLATFORM_CUDA_OPS_FLASHINFER_ATTENTION_H_
#define INFERX_PLATFORM_CUDA_OPS_FLASHINFER_ATTENTION_H_

#include "absl/status/status.h"

namespace inferx::cuda::ops {

// ADR 0029 rejects the recorded candidate pin for production because its
// required native-only, ahead-of-time closure is unavailable in this source
// checkout. This stable adapter probe makes the rejection explicit.
[[nodiscard]] absl::Status FlashInferAvailability();

}  // namespace inferx::cuda::ops

#endif  // INFERX_PLATFORM_CUDA_OPS_FLASHINFER_ATTENTION_H_
