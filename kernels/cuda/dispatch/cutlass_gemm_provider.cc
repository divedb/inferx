#include "cutlass_gemm_provider.h"

#include "cuda_tensor_checks.h"
#include "inferx/kernels/cuda/gemm_kernels.h"
#include "op_validation.h"

namespace inferx::kernels::cuda {
namespace {

constexpr uint16_t kMinCutlassComputeCapability = 80;

class CutlassGemmProviderBase {
 protected:
  static bool Available(const TensorView* primary, uint16_t compute_capability) {
    if (compute_capability != 0 && compute_capability < kMinCutlassComputeCapability) {
      return false;
    }
    return primary == nullptr || primary->shape().rank() == 2;
  }
};

class CutlassGemmProvider final : public GemmProvider, CutlassGemmProviderBase {
 public:
  ProviderId provider_id() const noexcept override { return ProviderId::kCutlass; }
  bool Available(const ops::GemmRequest* probe,
                 uint16_t compute_capability) const noexcept override {
    return probe == nullptr ||
           CutlassGemmProviderBase::Available(&probe->input, compute_capability);
  }
  absl::Status Launch(const ops::GemmRequest& request,
                      const CudaLaunchContext& context) const override {
    absl::Status status = ValidateGemmForCuda(request, context);
    if (!status.ok()) return status;
    return CheckLaunchResult(
        gemm::LaunchCutlassGemm(Address(request.input), Address(request.weight),
                                request.addend.has_value() ? Address(*request.addend) : nullptr,
                                Address(request.output), request.input.shape().dim(0),
                                request.input.shape().dim(1), request.output.shape().dim(1),
                                request.alpha, request.beta,
                                ToKernelStorageType(request.input.dtype()), context.stream),
        "kernels_cuda.gemm.cutlass.launch", context);
  }
};

class CutlassLogitsProvider final : public LogitsProvider, CutlassGemmProviderBase {
 public:
  ProviderId provider_id() const noexcept override { return ProviderId::kCutlass; }
  bool Available(const ops::LogitsRequest* probe,
                 uint16_t compute_capability) const noexcept override {
    return probe == nullptr ||
           CutlassGemmProviderBase::Available(&probe->hidden, compute_capability);
  }
  absl::Status Launch(const ops::LogitsRequest& request,
                      const CudaLaunchContext& context) const override {
    absl::Status status = ValidateLogitsForCuda(request, context);
    if (!status.ok()) return status;
    return CheckLaunchResult(
        gemm::LaunchCutlassGemm(Address(request.hidden), Address(request.weight), nullptr,
                                Address(request.output), request.hidden.shape().dim(0),
                                request.hidden.shape().dim(1), request.output.shape().dim(1),
                                /*alpha=*/1.0F, /*beta=*/0.0F,
                                ToKernelStorageType(request.hidden.dtype()), context.stream),
        "kernels_cuda.logits.cutlass.launch", context);
  }
};

}  // namespace

std::unique_ptr<GemmProvider> MakeCutlassGemmProvider() {
  return std::make_unique<CutlassGemmProvider>();
}
std::unique_ptr<LogitsProvider> MakeCutlassLogitsProvider() {
  return std::make_unique<CutlassLogitsProvider>();
}

}  // namespace inferx::kernels::cuda
