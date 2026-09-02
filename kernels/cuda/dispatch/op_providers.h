// SimpleOpProvider: binds a validated launch function (op_launchers.h) as a
// provider implementation. Factories cover the kernels-layer contracts
// (ADR 0031 taxonomy), including the real FlashInfer attention providers.
#ifndef INFERX_KERNELS_CUDA_DISPATCH_OP_PROVIDERS_H_
#define INFERX_KERNELS_CUDA_DISPATCH_OP_PROVIDERS_H_

#include <cstdint>
#include <memory>

#include "cuda_kernel_backend.h"

namespace inferx::kernels::cuda {

template <typename Request>
using OpLaunchFn = absl::Status (*)(const Request&, const CudaLaunchContext&);

template <typename Request>
using OpAvailableFn = bool (*)(const Request*, uint16_t);

template <typename Request>
class SimpleOpProvider final : public OpProvider<Request> {
 public:
  SimpleOpProvider(ProviderId id, OpLaunchFn<Request> launch, OpAvailableFn<Request> available)
      : id_(id), launch_(launch), available_(available) {}

  [[nodiscard]] ProviderId provider_id() const noexcept override { return id_; }
  [[nodiscard]] bool Available(const Request* probe,
                               uint16_t compute_capability) const noexcept override {
    return available_(probe, compute_capability);
  }
  [[nodiscard]] absl::Status Launch(const Request& request,
                                    const CudaLaunchContext& context) const override {
    return launch_(request, context);
  }

 private:
  ProviderId id_;
  OpLaunchFn<Request> launch_;
  OpAvailableFn<Request> available_;
};

// flashinfer-backed providers.
[[nodiscard]] std::unique_ptr<ActMulProvider> MakeFlashInferActMulProvider();
[[nodiscard]] std::unique_ptr<FusedAddRmsNormProvider> MakeFlashInferFusedAddRmsNormProvider();
[[nodiscard]] std::unique_ptr<GemmaRmsNormProvider> MakeFlashInferGemmaRmsNormProvider();
[[nodiscard]] std::unique_ptr<QkRmsNormProvider> MakeFlashInferQkRmsNormProvider();
[[nodiscard]] std::unique_ptr<TopPRenormProvider> MakeFlashInferTopPRenormProvider();

// owned (last-resort) providers.
[[nodiscard]] std::unique_ptr<Add3Provider> MakeOwnedAdd3Provider();
[[nodiscard]] std::unique_ptr<HadamardTransformProvider> MakeOwnedHadamardProvider();
[[nodiscard]] std::unique_ptr<ArgmaxProvider> MakeOwnedArgmaxProvider();
[[nodiscard]] std::unique_ptr<TopKRenormProvider> MakeOwnedTopKRenormProvider();
[[nodiscard]] std::unique_ptr<Fp8QuantProvider> MakeOwnedFp8QuantProvider();
[[nodiscard]] std::unique_ptr<SoftmaxTopKProvider> MakeOwnedSoftmaxTopKProvider();
[[nodiscard]] std::unique_ptr<SigmoidBiasTopKProvider> MakeOwnedSigmoidBiasTopKProvider();
[[nodiscard]] std::unique_ptr<AttnResProvider> MakeOwnedAttnResProvider();
[[nodiscard]] std::unique_ptr<HcMixProvider> MakeOwnedHcMixProvider();
[[nodiscard]] std::unique_ptr<HcCombineProvider> MakeOwnedHcCombineProvider();
[[nodiscard]] std::unique_ptr<MhcPreProvider> MakeOwnedMhcPreProvider();
[[nodiscard]] std::unique_ptr<MhcPostProvider> MakeOwnedMhcPostProvider();
[[nodiscard]] std::unique_ptr<RingSconvProvider> MakeOwnedRingSconvProvider();

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_DISPATCH_OP_PROVIDERS_H_
