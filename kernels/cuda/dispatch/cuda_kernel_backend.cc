#include "cuda_kernel_backend.h"

#include <utility>

#include "absl/strings/str_cat.h"
#include "custom_providers.h"
#include "hpc_ops_providers.h"

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
              absl::StrCat("kernels_cuda.", op_name, ": forced provider ",
                           ProviderIdName(*forced), " is unavailable for this request"));
        }
        return provider.get();
      }
    }
    return absl::UnimplementedError(absl::StrCat("kernels_cuda.", op_name, ": forced provider ",
                                                 ProviderIdName(*forced), " is not registered"));
  }
  for (ProviderId candidate : ProviderPreferenceChain()) {
    for (const auto& provider : providers) {
      if (provider->provider_id() == candidate &&
            provider->Available(probe, compute_capability)) {
        return provider.get();
      }
    }
  }
  return absl::UnimplementedError(absl::StrCat(
      "kernels_cuda.", op_name, ": no available provider for compute capability ",
      compute_capability));
}

absl::StatusOr<ProviderId> CudaKernelBackend::SelectProvider(
    ops::OpKind kind, uint16_t compute_capability, std::optional<ProviderId> forced) const {
  switch (kind) {
    case ops::OpKind::kEmbedding: {
      auto resolved =
          Resolve(embedding_, static_cast<const ops::EmbeddingRequest*>(nullptr), compute_capability, forced, "embedding");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kGemm: {
      auto resolved = Resolve(gemm_, static_cast<const ops::GemmRequest*>(nullptr), compute_capability, forced, "gemm");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kLogits: {
      auto resolved = Resolve(logits_, static_cast<const ops::LogitsRequest*>(nullptr), compute_capability, forced, "logits");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kRmsNorm: {
      auto resolved =
          Resolve(rms_norm_, static_cast<const ops::RmsNormRequest*>(nullptr), compute_capability, forced, "rms_norm");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kRope: {
      auto resolved = Resolve(rope_, static_cast<const ops::RopeRequest*>(nullptr), compute_capability, forced, "rope");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kSiluMultiply: {
      auto resolved =
          Resolve(swiglu_, static_cast<const ops::SwiGluRequest*>(nullptr), compute_capability, forced, "silu_multiply");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kResidual: {
      auto resolved =
          Resolve(residual_, static_cast<const ops::ResidualRequest*>(nullptr), compute_capability, forced, "residual");
      if (!resolved.ok()) return std::move(resolved).status();
      return (*resolved)->provider_id();
    }
    case ops::OpKind::kAttention: {
      auto resolved =
          Resolve(attention_, static_cast<const ops::AttentionRequest*>(nullptr), compute_capability, forced, "attention");
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

absl::Status CudaKernelBackend::LaunchAttention(
    const ops::AttentionRequest& request, const DeviceAttentionMetadata& metadata,
    const KernelExecutionContext& context, std::optional<ProviderId> forced) {
  absl::StatusOr<AttentionProvider*> provider =
      Resolve(attention_, &request, context.compute_capability, forced, "attention");
  if (!provider.ok()) return provider.status();
  return (*provider)->Launch(request, metadata, ToLaunchContext(context));
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
  backend->AddEmbeddingProvider(MakeOwnedEmbeddingProvider());
  backend->AddSwiGluProvider(MakeOwnedSwiGluProvider());
  backend->AddResidualProvider(MakeOwnedResidualProvider());
  return RegisterKernelBackend(std::move(backend));
}

}  // namespace inferx::kernels::cuda
