// FlashInfer providers (reuse tier 2 in the ADR 0031 chain): RMSNorm and
// RoPE wrap ahead-of-time instantiations of flashinfer device kernels; the
// attention provider records the pending paged-KV qualification gap.
// Compiled only when the flashinfer gitlink is initialized and the feature
// is enabled (INFERX_KERNELS_HAVE_FLASHINFER).
#ifndef INFERX_KERNELS_CUDA_DISPATCH_FLASHINFER_PROVIDERS_H_
#define INFERX_KERNELS_CUDA_DISPATCH_FLASHINFER_PROVIDERS_H_

#include <memory>

#include "cuda_kernel_backend.h"

namespace inferx::kernels::cuda {

[[nodiscard]] std::unique_ptr<RmsNormProvider> MakeFlashInferRmsNormProvider();
[[nodiscard]] std::unique_ptr<RopeProvider> MakeFlashInferRopeProvider();
[[nodiscard]] std::unique_ptr<AttentionProvider> MakeFlashInferAttentionProvider();

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_DISPATCH_FLASHINFER_PROVIDERS_H_
