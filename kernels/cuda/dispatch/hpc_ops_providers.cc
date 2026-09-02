#include "hpc_ops_providers.h"

namespace inferx::kernels::cuda {
namespace {

// hpc-ops builds one module per architecture and supports only
// {90, 100, 103} at the pin (HPC_KNOWN_ARCHS in its CMakeLists). Below SM90
// the provider is unavailable by construction, which is the common case on
// the SM89 runner.

class HpcOpsAttentionProvider final : public AttentionProvider {
 public:
  ProviderId provider_id() const noexcept override { return ProviderId::kHpcOps; }
  bool Available(const ops::AttentionRequest*, uint16_t /*compute_capability*/) const noexcept {
    // Even on SM90+ the attention kernels are warp-specialized builds behind
    // the hpc-ops module system; the InferX adapter is unqualified (ADR 0032).
    return false;
  }
  absl::Status Launch(const ops::AttentionRequest&, const DeviceAttentionMetadata&,
                      const CudaLaunchContext&) const override {
    return absl::UnimplementedError(
        "kernels_cuda.attention.hpc_ops: SM90+ module not integrated at the pin (ADR 0032)");
  }
};

class HpcOpsGemmProvider final : public GemmProvider {
 public:
  ProviderId provider_id() const noexcept override { return ProviderId::kHpcOps; }
  bool Available(const ops::GemmRequest*, uint16_t /*compute_capability*/) const noexcept {
    // hpc-ops GEMMs at the pin are FP8 group-GEMM with TMA descriptor
    // workspaces on SM90+; the InferX contract is FP32/FP16/BF16 dense.
    return false;
  }
  absl::Status Launch(const ops::GemmRequest&, const CudaLaunchContext&) const override {
    return absl::UnimplementedError(
        "kernels_cuda.gemm.hpc_ops: FP8/SM90+ group-GEMM does not match the dense "
        "FP32/FP16/BF16 contract (ADR 0032)");
  }
};

}  // namespace

std::unique_ptr<AttentionProvider> MakeHpcOpsAttentionProvider() {
  return std::make_unique<HpcOpsAttentionProvider>();
}
std::unique_ptr<GemmProvider> MakeHpcOpsGemmProvider() {
  return std::make_unique<HpcOpsGemmProvider>();
}

}  // namespace inferx::kernels::cuda
