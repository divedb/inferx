#include "op_providers.h"

#include "op_launchers.h"

namespace inferx::kernels::cuda {
namespace {

using launchers::ActMul;
using launchers::ActMulAvailable;
using launchers::Add3;
using launchers::Add3Available;
using launchers::Argmax;
using launchers::ArgmaxAvailable;
using launchers::AttnRes;
using launchers::AttnResAvailable;
using launchers::Fp8Quant;
using launchers::Fp8QuantAvailable;
using launchers::FusedAddRmsNorm;
using launchers::FusedAddRmsNormAvailable;
using launchers::GemmaRmsNorm;
using launchers::GemmaRmsNormAvailable;
using launchers::HadamardTransform;
using launchers::HadamardTransformAvailable;
using launchers::HcCombine;
using launchers::HcCombineAvailable;
using launchers::HcMix;
using launchers::HcMixAvailable;
using launchers::MhcPost;
using launchers::MhcPostAvailable;
using launchers::MhcPre;
using launchers::MhcPreAvailable;
using launchers::QkRmsNorm;
using launchers::QkRmsNormAvailable;
using launchers::RingSconv;
using launchers::RingSconvAvailable;
using launchers::SigmoidBiasTopK;
using launchers::SigmoidBiasTopKAvailable;
using launchers::SoftmaxTopK;
using launchers::SoftmaxTopKAvailable;
using launchers::TopKRenorm;
using launchers::TopKRenormAvailable;
using launchers::TopPRenorm;
using launchers::TopPRenormAvailable;

}  // namespace

std::unique_ptr<ActMulProvider> MakeFlashInferActMulProvider() {
  return std::make_unique<SimpleOpProvider<ActMulRequest>>(ProviderId::kFlashInfer, ActMul,
                                                           ActMulAvailable);
}
std::unique_ptr<FusedAddRmsNormProvider> MakeFlashInferFusedAddRmsNormProvider() {
  return std::make_unique<SimpleOpProvider<FusedAddRmsNormRequest>>(
      ProviderId::kFlashInfer, FusedAddRmsNorm, FusedAddRmsNormAvailable);
}
std::unique_ptr<GemmaRmsNormProvider> MakeFlashInferGemmaRmsNormProvider() {
  return std::make_unique<SimpleOpProvider<GemmaRmsNormRequest>>(
      ProviderId::kFlashInfer, GemmaRmsNorm, GemmaRmsNormAvailable);
}
std::unique_ptr<QkRmsNormProvider> MakeFlashInferQkRmsNormProvider() {
  return std::make_unique<SimpleOpProvider<QkRmsNormRequest>>(ProviderId::kFlashInfer, QkRmsNorm,
                                                              QkRmsNormAvailable);
}
std::unique_ptr<TopPRenormProvider> MakeFlashInferTopPRenormProvider() {
  return std::make_unique<SimpleOpProvider<TopPRenormRequest>>(ProviderId::kFlashInfer,
                                                               TopPRenorm, TopPRenormAvailable);
}

std::unique_ptr<Add3Provider> MakeOwnedAdd3Provider() {
  return std::make_unique<SimpleOpProvider<Add3Request>>(ProviderId::kInferxOwned, Add3,
                                                         Add3Available);
}
std::unique_ptr<HadamardTransformProvider> MakeOwnedHadamardProvider() {
  return std::make_unique<SimpleOpProvider<HadamardTransformRequest>>(
      ProviderId::kInferxOwned, HadamardTransform, HadamardTransformAvailable);
}
std::unique_ptr<ArgmaxProvider> MakeOwnedArgmaxProvider() {
  return std::make_unique<SimpleOpProvider<ArgmaxRequest>>(ProviderId::kInferxOwned, Argmax,
                                                           ArgmaxAvailable);
}
std::unique_ptr<TopKRenormProvider> MakeOwnedTopKRenormProvider() {
  return std::make_unique<SimpleOpProvider<TopKRenormRequest>>(ProviderId::kInferxOwned,
                                                               TopKRenorm, TopKRenormAvailable);
}
std::unique_ptr<Fp8QuantProvider> MakeOwnedFp8QuantProvider() {
  return std::make_unique<SimpleOpProvider<Fp8QuantRequest>>(ProviderId::kInferxOwned, Fp8Quant,
                                                             Fp8QuantAvailable);
}
std::unique_ptr<SoftmaxTopKProvider> MakeOwnedSoftmaxTopKProvider() {
  return std::make_unique<SimpleOpProvider<SoftmaxTopKRequest>>(ProviderId::kInferxOwned,
                                                                SoftmaxTopK,
                                                                SoftmaxTopKAvailable);
}
std::unique_ptr<SigmoidBiasTopKProvider> MakeOwnedSigmoidBiasTopKProvider() {
  return std::make_unique<SimpleOpProvider<SigmoidBiasTopKRequest>>(
      ProviderId::kInferxOwned, SigmoidBiasTopK, SigmoidBiasTopKAvailable);
}
std::unique_ptr<AttnResProvider> MakeOwnedAttnResProvider() {
  return std::make_unique<SimpleOpProvider<AttnResRequest>>(ProviderId::kInferxOwned, AttnRes,
                                                            AttnResAvailable);
}
std::unique_ptr<HcMixProvider> MakeOwnedHcMixProvider() {
  return std::make_unique<SimpleOpProvider<HcMixRequest>>(ProviderId::kInferxOwned, HcMix,
                                                          HcMixAvailable);
}
std::unique_ptr<HcCombineProvider> MakeOwnedHcCombineProvider() {
  return std::make_unique<SimpleOpProvider<HcCombineRequest>>(ProviderId::kInferxOwned, HcCombine,
                                                              HcCombineAvailable);
}
std::unique_ptr<MhcPreProvider> MakeOwnedMhcPreProvider() {
  return std::make_unique<SimpleOpProvider<MhcPreRequest>>(ProviderId::kInferxOwned, MhcPre,
                                                           MhcPreAvailable);
}
std::unique_ptr<MhcPostProvider> MakeOwnedMhcPostProvider() {
  return std::make_unique<SimpleOpProvider<MhcPostRequest>>(ProviderId::kInferxOwned, MhcPost,
                                                            MhcPostAvailable);
}
std::unique_ptr<RingSconvProvider> MakeOwnedRingSconvProvider() {
  return std::make_unique<SimpleOpProvider<RingSconvRequest>>(ProviderId::kInferxOwned, RingSconv,
                                                              RingSconvAvailable);
}

}  // namespace inferx::kernels::cuda
