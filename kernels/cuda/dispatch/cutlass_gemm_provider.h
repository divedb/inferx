// CUTLASS GEMM/logits provider (last-resort tier; justification in ADR 0032).
// Compiled only when the cutlass gitlink is initialized and the feature is
// enabled (INFERX_KERNELS_HAVE_CUTLASS).
#ifndef INFERX_KERNELS_CUDA_DISPATCH_CUTLASS_GEMM_PROVIDER_H_
#define INFERX_KERNELS_CUDA_DISPATCH_CUTLASS_GEMM_PROVIDER_H_

#include <memory>

#include "cuda_kernel_backend.h"

namespace inferx::kernels::cuda {

[[nodiscard]] std::unique_ptr<GemmProvider> MakeCutlassGemmProvider();
[[nodiscard]] std::unique_ptr<LogitsProvider> MakeCutlassLogitsProvider();

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_DISPATCH_CUTLASS_GEMM_PROVIDER_H_
