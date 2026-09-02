#include "cuda_kernel_backend.h"

#include <utility>

#include "absl/strings/str_cat.h"
#include "custom_providers.h"
#include "hpc_ops_providers.h"
#include "op_providers.h"

#if defined(INFERX_KERNELS_HAVE_CUTLASS)
#include "cutlass_gemm_provider.h"
#endif
#if defined(INFERX_KERNELS_HAVE_FLASHINFER)
#include "flashinfer_providers.h"
#endif

namespace inferx::kernels::cuda {
namespace {

CudaLaunchContext ToLaunchContext(const KernelExecutionContext& context) {
  return CudaLaunchContext{
      /*stream=*/reinterpret_cast<cudaStream_t>(context.stream_handle),
      /*device=*/context.device.ordinal,
      /*compute_capability=*/context.compute_capability,
      /*observer=*/context.observer,
      /*workspace=*/context.workspace,
      /*workspace_bytes=*/context.workspace_bytes,
  };
}

}  // namespace

void CudaKernelBackend::AddEmbeddingProvider(std::unique_ptr<EmbeddingProvider> provider) {
  embedding_.push_back(std::move(provider));
}
void CudaKernelBackend::AddGemmProvider(std::unique_ptr<GemmProvider> provider) {
  gemm_.push_back(std::move(provider));
}
void CudaKernelBackend::AddLogitsProvider(std::unique_ptr<LogitsProvider> provider) {
  logits_.push_back(std::move(provider));
}
void CudaKernelBackend::AddRmsNormProvider(std::unique_ptr<RmsNormProvider> provider) {
  rms_norm_.push_back(std::move(provider));
}
void CudaKernelBackend::AddRopeProvider(std::unique_ptr<RopeProvider> provider) {
  rope_.push_back(std::move(provider));
}
void CudaKernelBackend::AddSwiGluProvider(std::unique_ptr<SwiGluProvider> provider) {
  swiglu_.push_back(std::move(provider));
}
void CudaKernelBackend::AddResidualProvider(std::unique_ptr<ResidualProvider> provider) {
  residual_.push_back(std::move(provider));
}
void CudaKernelBackend::AddAttentionProvider(std::unique_ptr<AttentionProvider> provider) {
  attention_.push_back(std::move(provider));
}
void CudaKernelBackend::AddActMulProvider(std::unique_ptr<ActMulProvider> provider) {
  act_mul_.push_back(std::move(provider));
}
void CudaKernelBackend::AddAdd3Provider(std::unique_ptr<Add3Provider> provider) {
  add3_.push_back(std::move(provider));
}
void CudaKernelBackend::AddFusedAddRmsNormProvider(
    std::unique_ptr<FusedAddRmsNormProvider> provider) {
  fused_add_rms_norm_.push_back(std::move(provider));
}
void CudaKernelBackend::AddGemmaRmsNormProvider(std::unique_ptr<GemmaRmsNormProvider> provider) {
  gemma_rms_norm_.push_back(std::move(provider));
}
void CudaKernelBackend::AddQkRmsNormProvider(std::unique_ptr<QkRmsNormProvider> provider) {
  qk_rms_norm_.push_back(std::move(provider));
}
void CudaKernelBackend::AddHadamardTransformProvider(
    std::unique_ptr<HadamardTransformProvider> provider) {
  hadamard_.push_back(std::move(provider));
}
void CudaKernelBackend::AddArgmaxProvider(std::unique_ptr<ArgmaxProvider> provider) {
  argmax_.push_back(std::move(provider));
}
void CudaKernelBackend::AddTopPRenormProvider(std::unique_ptr<TopPRenormProvider> provider) {
  top_p_renorm_.push_back(std::move(provider));
}
void CudaKernelBackend::AddTopKRenormProvider(std::unique_ptr<TopKRenormProvider> provider) {
  top_k_renorm_.push_back(std::move(provider));
}
void CudaKernelBackend::AddFp8QuantProvider(std::unique_ptr<Fp8QuantProvider> provider) {
  fp8_quant_.push_back(std::move(provider));
}
void CudaKernelBackend::AddSoftmaxTopKProvider(std::unique_ptr<SoftmaxTopKProvider> provider) {
  softmax_topk_.push_back(std::move(provider));
}
void CudaKernelBackend::AddSigmoidBiasTopKProvider(
    std::unique_ptr<SigmoidBiasTopKProvider> provider) {
  sigmoid_bias_topk_.push_back(std::move(provider));
}
void CudaKernelBackend::AddAttnResProvider(std::unique_ptr<AttnResProvider> provider) {
  attn_res_.push_back(std::move(provider));
}
void CudaKernelBackend::AddHcMixProvider(std::unique_ptr<HcMixProvider> provider) {
  hc_mix_.push_back(std::move(provider));
}
void CudaKernelBackend::AddHcCombineProvider(std::unique_ptr<HcCombineProvider> provider) {
  hc_combine_.push_back(std::move(provider));
}
void CudaKernelBackend::AddMhcPreProvider(std::unique_ptr<MhcPreProvider> provider) {
  mhc_pre_.push_back(std::move(provider));
}
void CudaKernelBackend::AddMhcPostProvider(std::unique_ptr<MhcPostProvider> provider) {
  mhc_post_.push_back(std::move(provider));
}
void CudaKernelBackend::AddRingSconvProvider(std::unique_ptr<RingSconvProvider> provider) {
  ring_sconv_.push_back(std::move(provider));
}

// Resolves a launch through the preference chain. A null probe queries
// chain-level availability only; providers treat shape constraints as absent
// in that case.
template <typename Provider, typename Request>
absl::StatusOr<Provider*> CudaKernelBackend::Resolve(
    const std::vector<std::unique_ptr<Provider>>& providers, const Request* probe,
    uint16_t compute_capability, std::optional<ProviderId> forced, const char* op_name) const {
  if (forced.has_value()) {
    for (const auto& provider : providers) {
      if (provider->provider_id() == *forced) {
        if (!provider->Available(probe, compute_capability)) {
          return absl::UnimplementedError(
              absl::StrCat("kernels_cuda.", op_name, ": forced provider ", ProviderIdName(*forced),
                           " is unavailable for this request"));
        }
        return provider.get();
      }
    }
    return absl::UnimplementedError(absl::StrCat("kernels_cuda.", op_name, ": forced provider ",
                                                 ProviderIdName(*forced), " is not registered"));
  }
  for (ProviderId candidate : ProviderPreferenceChain()) {
    for (const auto& provider : providers) {
      if (provider->provider_id() == candidate && provider->Available(probe, compute_capability)) {
        return provider.get();
      }
    }
  }
  return absl::UnimplementedError(absl::StrCat("kernels_cuda.", op_name,
                                               ": no available provider for compute capability ",
                                               compute_capability));
}

absl::StatusOr<ProviderId> CudaKernelBackend::SelectProvider(
    ops::OpKind kind, uint16_t compute_capability, std::optional<ProviderId> forced) const {
  switch (kind) {
    case ops::OpKind::kEmbedding: {
      auto resolved = Resolve(embedding_, static_cast<const ops::EmbeddingRequest*>(nullptr),
                              compute_capability, forced, "embedding");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kGemm: {
      auto resolved = Resolve(gemm_, static_cast<const ops::GemmRequest*>(nullptr),
                              compute_capability, forced, "gemm");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kLogits: {
      auto resolved = Resolve(logits_, static_cast<const ops::LogitsRequest*>(nullptr),
                              compute_capability, forced, "logits");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kRmsNorm: {
      auto resolved = Resolve(rms_norm_, static_cast<const ops::RmsNormRequest*>(nullptr),
                              compute_capability, forced, "rms_norm");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kRope: {
      auto resolved = Resolve(rope_, static_cast<const ops::RopeRequest*>(nullptr),
                              compute_capability, forced, "rope");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kSiluMultiply: {
      auto resolved = Resolve(swiglu_, static_cast<const ops::SwiGluRequest*>(nullptr),
                              compute_capability, forced, "silu_multiply");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kResidual: {
      auto resolved = Resolve(residual_, static_cast<const ops::ResidualRequest*>(nullptr),
                              compute_capability, forced, "residual");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kAttention: {
      auto resolved = Resolve(attention_, static_cast<const ops::AttentionRequest*>(nullptr),
                              compute_capability, forced, "attention");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    default:
      return absl::UnimplementedError("kernels_cuda.select: op has no CUDA provider surface yet");
  }
}

absl::Status CudaKernelBackend::LaunchEmbedding(const ops::EmbeddingRequest& request,
                                                const KernelExecutionContext& context,
                                                std::optional<ProviderId> forced) {
  absl::StatusOr<EmbeddingProvider*> provider =
      Resolve(embedding_, &request, context.compute_capability, forced, "embedding");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}

absl::Status CudaKernelBackend::LaunchGemm(const ops::GemmRequest& request,
                                           const KernelExecutionContext& context,
                                           std::optional<ProviderId> forced) {
  absl::StatusOr<GemmProvider*> provider =
      Resolve(gemm_, &request, context.compute_capability, forced, "gemm");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}

absl::Status CudaKernelBackend::LaunchLogits(const ops::LogitsRequest& request,
                                             const KernelExecutionContext& context,
                                             std::optional<ProviderId> forced) {
  absl::StatusOr<LogitsProvider*> provider =
      Resolve(logits_, &request, context.compute_capability, forced, "logits");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}

absl::Status CudaKernelBackend::LaunchRmsNorm(const ops::RmsNormRequest& request,
                                              const KernelExecutionContext& context,
                                              std::optional<ProviderId> forced) {
  absl::StatusOr<RmsNormProvider*> provider =
      Resolve(rms_norm_, &request, context.compute_capability, forced, "rms_norm");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}

absl::Status CudaKernelBackend::LaunchRope(const ops::RopeRequest& request,
                                           const KernelExecutionContext& context,
                                           std::optional<ProviderId> forced) {
  absl::StatusOr<RopeProvider*> provider =
      Resolve(rope_, &request, context.compute_capability, forced, "rope");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}

absl::Status CudaKernelBackend::LaunchSwiGlu(const ops::SwiGluRequest& request,
                                             const KernelExecutionContext& context,
                                             std::optional<ProviderId> forced) {
  absl::StatusOr<SwiGluProvider*> provider =
      Resolve(swiglu_, &request, context.compute_capability, forced, "silu_multiply");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}

absl::Status CudaKernelBackend::LaunchResidual(const ops::ResidualRequest& request,
                                               const KernelExecutionContext& context,
                                               std::optional<ProviderId> forced) {
  absl::StatusOr<ResidualProvider*> provider =
      Resolve(residual_, &request, context.compute_capability, forced, "residual");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}

absl::Status CudaKernelBackend::LaunchAttention(const ops::AttentionRequest& request,
                                                const DeviceAttentionMetadata& metadata,
                                                const KernelExecutionContext& context,
                                                std::optional<ProviderId> forced) {
  absl::StatusOr<AttentionProvider*> provider =
      Resolve(attention_, &request, context.compute_capability, forced, "attention");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, metadata, ToLaunchContext(context));
}

absl::Status CudaKernelBackend::LaunchActMul(const ActMulRequest& request,
                                             const KernelExecutionContext& context,
                                             std::optional<ProviderId> forced) {
  absl::StatusOr<ActMulProvider*> provider =
      Resolve(act_mul_, &request, context.compute_capability, forced, "act_mul");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchAdd3(const Add3Request& request,
                                           const KernelExecutionContext& context,
                                           std::optional<ProviderId> forced) {
  absl::StatusOr<Add3Provider*> provider =
      Resolve(add3_, &request, context.compute_capability, forced, "add3");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchFusedAddRmsNorm(const FusedAddRmsNormRequest& request,
                                                      const KernelExecutionContext& context,
                                                      std::optional<ProviderId> forced) {
  absl::StatusOr<FusedAddRmsNormProvider*> provider = Resolve(
      fused_add_rms_norm_, &request, context.compute_capability, forced, "fused_add_rms_norm");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchGemmaRmsNorm(const GemmaRmsNormRequest& request,
                                                   const KernelExecutionContext& context,
                                                   std::optional<ProviderId> forced) {
  absl::StatusOr<GemmaRmsNormProvider*> provider =
      Resolve(gemma_rms_norm_, &request, context.compute_capability, forced, "gemma_rms_norm");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchQkRmsNorm(const QkRmsNormRequest& request,
                                                const KernelExecutionContext& context,
                                                std::optional<ProviderId> forced) {
  absl::StatusOr<QkRmsNormProvider*> provider =
      Resolve(qk_rms_norm_, &request, context.compute_capability, forced, "qk_rms_norm");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchHadamardTransform(const HadamardTransformRequest& request,
                                                        const KernelExecutionContext& context,
                                                        std::optional<ProviderId> forced) {
  absl::StatusOr<HadamardTransformProvider*> provider =
      Resolve(hadamard_, &request, context.compute_capability, forced, "hadamard_transform");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchArgmax(const ArgmaxRequest& request,
                                             const KernelExecutionContext& context,
                                             std::optional<ProviderId> forced) {
  absl::StatusOr<ArgmaxProvider*> provider =
      Resolve(argmax_, &request, context.compute_capability, forced, "argmax");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchTopPRenorm(const TopPRenormRequest& request,
                                                 const KernelExecutionContext& context,
                                                 std::optional<ProviderId> forced) {
  absl::StatusOr<TopPRenormProvider*> provider =
      Resolve(top_p_renorm_, &request, context.compute_capability, forced, "top_p_renorm");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchTopKRenorm(const TopKRenormRequest& request,
                                                 const KernelExecutionContext& context,
                                                 std::optional<ProviderId> forced) {
  absl::StatusOr<TopKRenormProvider*> provider =
      Resolve(top_k_renorm_, &request, context.compute_capability, forced, "top_k_renorm");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchFp8Quant(const Fp8QuantRequest& request,
                                               const KernelExecutionContext& context,
                                               std::optional<ProviderId> forced) {
  absl::StatusOr<Fp8QuantProvider*> provider =
      Resolve(fp8_quant_, &request, context.compute_capability, forced, "fp8_quant");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchSoftmaxTopK(const SoftmaxTopKRequest& request,
                                                  const KernelExecutionContext& context,
                                                  std::optional<ProviderId> forced) {
  absl::StatusOr<SoftmaxTopKProvider*> provider =
      Resolve(softmax_topk_, &request, context.compute_capability, forced, "softmax_top_k");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchSigmoidBiasTopK(const SigmoidBiasTopKRequest& request,
                                                      const KernelExecutionContext& context,
                                                      std::optional<ProviderId> forced) {
  absl::StatusOr<SigmoidBiasTopKProvider*> provider = Resolve(
      sigmoid_bias_topk_, &request, context.compute_capability, forced, "sigmoid_bias_top_k");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchAttnRes(const AttnResRequest& request,
                                              const KernelExecutionContext& context,
                                              std::optional<ProviderId> forced) {
  absl::StatusOr<AttnResProvider*> provider =
      Resolve(attn_res_, &request, context.compute_capability, forced, "attn_res");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchHcMix(const HcMixRequest& request,
                                            const KernelExecutionContext& context,
                                            std::optional<ProviderId> forced) {
  absl::StatusOr<HcMixProvider*> provider =
      Resolve(hc_mix_, &request, context.compute_capability, forced, "hc_mix");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchHcCombine(const HcCombineRequest& request,
                                                const KernelExecutionContext& context,
                                                std::optional<ProviderId> forced) {
  absl::StatusOr<HcCombineProvider*> provider =
      Resolve(hc_combine_, &request, context.compute_capability, forced, "hc_combine");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchMhcPre(const MhcPreRequest& request,
                                             const KernelExecutionContext& context,
                                             std::optional<ProviderId> forced) {
  absl::StatusOr<MhcPreProvider*> provider =
      Resolve(mhc_pre_, &request, context.compute_capability, forced, "mhc_pre");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchMhcPost(const MhcPostRequest& request,
                                              const KernelExecutionContext& context,
                                              std::optional<ProviderId> forced) {
  absl::StatusOr<MhcPostProvider*> provider =
      Resolve(mhc_post_, &request, context.compute_capability, forced, "mhc_post");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}
absl::Status CudaKernelBackend::LaunchRingSconv(const RingSconvRequest& request,
                                                const KernelExecutionContext& context,
                                                std::optional<ProviderId> forced) {
  absl::StatusOr<RingSconvProvider*> provider =
      Resolve(ring_sconv_, &request, context.compute_capability, forced, "ring_sconv");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, ToLaunchContext(context));
}

absl::Status RegisterCudaKernelBackend() {
  auto backend = std::make_unique<CudaKernelBackend>();
  // Preference chain order is fixed (hpc-ops -> flashinfer -> cutlass ->
  // vendor/owned); registration order does not matter. Providers compiled
  // into this build register their real availability probes; the ones whose
  // options/gitlinks are off are simply absent from the chain. The hpc-ops
  // probes always compile: they carry the provider-audit availability answer
  // (ADR 0032) without linking hpc-ops code.
  backend->AddAttentionProvider(MakeHpcOpsAttentionProvider());
  backend->AddGemmProvider(MakeHpcOpsGemmProvider());
#if defined(INFERX_KERNELS_HAVE_FLASHINFER)
  backend->AddRmsNormProvider(MakeFlashInferRmsNormProvider());
  backend->AddRopeProvider(MakeFlashInferRopeProvider());
  backend->AddAttentionProvider(MakeFlashInferAttentionProvider());
#endif
#if defined(INFERX_KERNELS_HAVE_CUTLASS)
  backend->AddGemmProvider(MakeCutlassGemmProvider());
  backend->AddLogitsProvider(MakeCutlassLogitsProvider());
#endif
#if defined(INFERX_KERNELS_HAVE_FLASHINFER)
  backend->AddActMulProvider(MakeFlashInferActMulProvider());
  backend->AddFusedAddRmsNormProvider(MakeFlashInferFusedAddRmsNormProvider());
  backend->AddGemmaRmsNormProvider(MakeFlashInferGemmaRmsNormProvider());
  backend->AddQkRmsNormProvider(MakeFlashInferQkRmsNormProvider());
  backend->AddTopPRenormProvider(MakeFlashInferTopPRenormProvider());
#endif
  backend->AddAdd3Provider(MakeOwnedAdd3Provider());
  backend->AddHadamardTransformProvider(MakeOwnedHadamardProvider());
  backend->AddArgmaxProvider(MakeOwnedArgmaxProvider());
  backend->AddTopKRenormProvider(MakeOwnedTopKRenormProvider());
  backend->AddFp8QuantProvider(MakeOwnedFp8QuantProvider());
  backend->AddSoftmaxTopKProvider(MakeOwnedSoftmaxTopKProvider());
  backend->AddSigmoidBiasTopKProvider(MakeOwnedSigmoidBiasTopKProvider());
  backend->AddAttnResProvider(MakeOwnedAttnResProvider());
  backend->AddHcMixProvider(MakeOwnedHcMixProvider());
  backend->AddHcCombineProvider(MakeOwnedHcCombineProvider());
  backend->AddMhcPreProvider(MakeOwnedMhcPreProvider());
  backend->AddMhcPostProvider(MakeOwnedMhcPostProvider());
  backend->AddRingSconvProvider(MakeOwnedRingSconvProvider());
  backend->AddEmbeddingProvider(MakeOwnedEmbeddingProvider());
  backend->AddSwiGluProvider(MakeOwnedSwiGluProvider());
  backend->AddResidualProvider(MakeOwnedResidualProvider());
  return RegisterKernelBackend(std::move(backend));
}

}  // namespace inferx::kernels::cuda
