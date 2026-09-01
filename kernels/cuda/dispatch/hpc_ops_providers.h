// hpc-ops providers (reuse tier 1 in the ADR 0031 chain).
//
// These are availability probes carrying the provider-audit answer recorded
// in ADR 0032 for the pinned revision; they link no hpc-ops code. The moment
// a qualified hpc-ops adapter lands (SM90+ hardware), these probes flip and
// the chain routes to hpc-ops first — no upper-layer change required.
#ifndef INFERX_KERNELS_CUDA_DISPATCH_HPC_OPS_PROVIDERS_H_
#define INFERX_KERNELS_CUDA_DISPATCH_HPC_OPS_PROVIDERS_H_

#include <memory>

#include "cuda_kernel_backend.h"

namespace inferx::kernels::cuda {

[[nodiscard]] std::unique_ptr<AttentionProvider> MakeHpcOpsAttentionProvider();
[[nodiscard]] std::unique_ptr<GemmProvider> MakeHpcOpsGemmProvider();

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_DISPATCH_HPC_OPS_PROVIDERS_H_
