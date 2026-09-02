// Custom (last-resort) providers for operators with no external coverage at
// the pinned revisions (ADR 0032 audit): embedding, silu_multiply, residual.
#ifndef INFERX_KERNELS_CUDA_DISPATCH_CUSTOM_PROVIDERS_H_
#define INFERX_KERNELS_CUDA_DISPATCH_CUSTOM_PROVIDERS_H_

#include <memory>

#include "cuda_kernel_backend.h"

namespace inferx::kernels::cuda {

[[nodiscard]] std::unique_ptr<EmbeddingProvider> MakeOwnedEmbeddingProvider();
[[nodiscard]] std::unique_ptr<SwiGluProvider> MakeOwnedSwiGluProvider();
[[nodiscard]] std::unique_ptr<ResidualProvider> MakeOwnedResidualProvider();

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_DISPATCH_CUSTOM_PROVIDERS_H_
