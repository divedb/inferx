#include "flashinfer_providers.h"

#include "cuda_tensor_checks.h"
#include "op_validation.h"
#include "inferx/kernels/cuda/layernorm_kernels.h"
#include "inferx/kernels/cuda/transform_kernels.h"

namespace inferx::kernels::cuda {
namespace {

constexpr uint16_t kMinFlashInferComputeCapability = 80;  // sm80 device kernels

class FlashInferRmsNormProvider final : public RmsNormProvider {
 public:
  ProviderId provider_id() const noexcept override { return ProviderId::kFlashInfer; }
  bool Available(const ops::RmsNormRequest* probe,
                 uint16_t compute_capability) const noexcept override {
    if (compute_capability != 0 && compute_capability < kMinFlashInferComputeCapability) {
      return false;
    }
    if (probe == nullptr) return true;  // chain probe: no shape constraints
    const uint64_t hidden = probe->input.shape().dim(1);
    const uint32_t vec = probe->input.dtype() == DType::kFloat32 ? 4 : 8;
    return hidden % vec == 0;
  }
  absl::Status Launch(const ops::RmsNormRequest& request,
                      const CudaLaunchContext& context) const override {
    absl::Status status = ValidateRmsNormForCuda(request, context);
    if (!status.ok()) return status;
    return CheckLaunchResult(
        layernorm::LaunchFlashInferRmsNorm(
            Address(request.input), Address(request.weight), Address(request.output),
            request.input.shape().dim(0), request.input.shape().dim(1), request.epsilon,
            ToKernelStorageType(request.input.dtype()), context.stream),
        "kernels_cuda.rms_norm.flashinfer.launch", context);
  }
};

class FlashInferRopeProvider final : public RopeProvider {
 public:
  ProviderId provider_id() const noexcept override { return ProviderId::kFlashInfer; }
  bool Available(const ops::RopeRequest* probe,
                 uint16_t compute_capability) const noexcept override {
    if (compute_capability != 0 && compute_capability < kMinFlashInferComputeCapability) {
      return false;
    }
    if (probe == nullptr) return true;  // chain probe: no shape constraints
    const uint64_t head_dimension = probe->query.shape().dim(2);
    return head_dimension == 32 || head_dimension == 64 || head_dimension == 128 ||
           head_dimension == 256;
  }
  absl::Status Launch(const ops::RopeRequest& request,
                      const CudaLaunchContext& context) const override {
    absl::Status status = ValidateRopeForCuda(request, context);
    if (!status.ok()) return status;
    return CheckLaunchResult(
        transform::LaunchFlashInferRope(
            Address(request.query), Address(request.key),
            static_cast<const int32_t*>(Address(request.positions)), Address(request.query_output),
            Address(request.key_output), request.query.shape().dim(0),
            request.query.shape().dim(1), request.key.shape().dim(1),
            request.query.shape().dim(2), request.theta,
            ToKernelStorageType(request.query.dtype()), context.stream),
        "kernels_cuda.rope.flashinfer.launch", context);
  }
};

// The pinned flashinfer attention path (decode.cuh/prefill.cuh) is
// paged-KV based: mapping the InferX contiguous-BSHD cache onto paged_kv_t
// plus scheduler params is a pending qualification item (ADR 0032). The
// provider stays registered — first in the chain after hpc-ops — and
// reports unavailable until that work lands, so the chain never silently
// skips it.
class FlashInferAttentionProvider final : public AttentionProvider {
 public:
  ProviderId provider_id() const noexcept override { return ProviderId::kFlashInfer; }
  bool Available(const ops::AttentionRequest*, uint16_t) const noexcept override {
    return false;
  }
  absl::Status Launch(const ops::AttentionRequest&, const DeviceAttentionMetadata&,
                      const CudaLaunchContext&) const override {
    return absl::UnimplementedError(
        "kernels_cuda.attention.flashinfer: paged-KV params mapping pending (ADR 0032)");
  }
};

}  // namespace

std::unique_ptr<RmsNormProvider> MakeFlashInferRmsNormProvider() {
  return std::make_unique<FlashInferRmsNormProvider>();
}
std::unique_ptr<RopeProvider> MakeFlashInferRopeProvider() {
  return std::make_unique<FlashInferRopeProvider>();
}
std::unique_ptr<AttentionProvider> MakeFlashInferAttentionProvider() {
  return std::make_unique<FlashInferAttentionProvider>();
}

}  // namespace inferx::kernels::cuda
