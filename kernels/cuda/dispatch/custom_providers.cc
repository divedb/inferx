#include "custom_providers.h"

#include "absl/status/statusor.h"
#include "cuda_tensor_checks.h"
#include "inferx/kernels/cuda/activation_kernels.h"
#include "inferx/kernels/cuda/embedding_kernels.h"
#include "op_validation.h"

namespace inferx::kernels::cuda {
namespace {

class OwnedEmbeddingProvider final : public EmbeddingProvider {
 public:
  ProviderId provider_id() const noexcept override { return ProviderId::kInferxOwned; }
  bool Available(const ops::EmbeddingRequest*, uint16_t) const noexcept override { return true; }
  absl::Status Launch(const ops::EmbeddingRequest& request,
                      const CudaLaunchContext& context) const override {
    absl::Status status = ValidateEmbeddingForCuda(request, context);
    if (!status.ok()) return status;
    return CheckLaunchResult(
        embedding::LaunchEmbeddingKernel(
            static_cast<const int32_t*>(Address(request.token_ids)), Address(request.weight),
            Address(request.output), request.token_ids.shape().dim(0),
            request.weight.shape().dim(0), request.weight.shape().dim(1),
            ToKernelStorageType(request.weight.dtype()), context.stream),
        "kernels_cuda.embedding.launch", context);
  }
};

class OwnedSwiGluProvider final : public SwiGluProvider {
 public:
  ProviderId provider_id() const noexcept override { return ProviderId::kInferxOwned; }
  bool Available(const ops::SwiGluRequest*, uint16_t) const noexcept override { return true; }
  absl::Status Launch(const ops::SwiGluRequest& request,
                      const CudaLaunchContext& context) const override {
    absl::Status status = ValidateSwiGluForCuda(request, context);
    if (!status.ok()) return status;
    absl::StatusOr<uint64_t> elements = request.left.shape().NumElements();
    if (!elements.ok()) return elements.status();
    return CheckLaunchResult(
        activation::LaunchSwiGluKernel(Address(request.left), Address(request.right),
                                       Address(request.output), *elements,
                                       ToKernelStorageType(request.left.dtype()), context.stream),
        "kernels_cuda.silu_multiply.launch", context);
  }
};

class OwnedResidualProvider final : public ResidualProvider {
 public:
  ProviderId provider_id() const noexcept override { return ProviderId::kInferxOwned; }
  bool Available(const ops::ResidualRequest*, uint16_t) const noexcept override { return true; }
  absl::Status Launch(const ops::ResidualRequest& request,
                      const CudaLaunchContext& context) const override {
    absl::Status status = ValidateResidualForCuda(request, context);
    if (!status.ok()) return status;
    absl::StatusOr<uint64_t> elements = request.left.shape().NumElements();
    if (!elements.ok()) return elements.status();
    return CheckLaunchResult(
        activation::LaunchResidualKernel(Address(request.left), Address(request.right),
                                         Address(request.output), *elements,
                                         ToKernelStorageType(request.left.dtype()), context.stream),
        "kernels_cuda.residual.launch", context);
  }
};

}  // namespace

std::unique_ptr<EmbeddingProvider> MakeOwnedEmbeddingProvider() {
  return std::make_unique<OwnedEmbeddingProvider>();
}
std::unique_ptr<SwiGluProvider> MakeOwnedSwiGluProvider() {
  return std::make_unique<OwnedSwiGluProvider>();
}
std::unique_ptr<ResidualProvider> MakeOwnedResidualProvider() {
  return std::make_unique<OwnedResidualProvider>();
}

}  // namespace inferx::kernels::cuda
