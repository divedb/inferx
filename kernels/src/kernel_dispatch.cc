#include "inferx/kernels/kernel_dispatch.h"

#include <array>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/kernels/ops/activation.h"
#include "inferx/kernels/ops/attention.h"
#include "inferx/kernels/ops/embedding.h"
#include "inferx/kernels/ops/gemm.h"
#include "inferx/kernels/ops/layernorm.h"
#include "inferx/kernels/ops/model_fused.h"
#include "inferx/kernels/ops/sampling.h"
#include "inferx/kernels/ops/transform.h"

namespace inferx::kernels {
namespace {

constexpr size_t kBackendSlotCount = 8;  // DeviceKind values are small uint8_t.

std::array<std::unique_ptr<KernelBackend>, kBackendSlotCount>& Backends() noexcept {
  static std::array<std::unique_ptr<KernelBackend>, kBackendSlotCount> backends{};
  return backends;
}

constexpr ProviderId kPreferenceChain[] = {
    ProviderId::kHpcOps,   ProviderId::kFlashInfer,  ProviderId::kCutlass,
    ProviderId::kCublasLt, ProviderId::kInferxOwned,
};

absl::StatusOr<KernelBackend*> BackendFor(const KernelExecutionContext& context) {
  const auto kind = static_cast<uint8_t>(context.device.kind);
  if (kind >= kBackendSlotCount) {
    return absl::InvalidArgumentError("kernels: device kind out of range for kernel dispatch");
  }
  KernelBackend* backend = Backends()[kind].get();
  if (backend == nullptr) {
    return absl::UnimplementedError(
        absl::StrCat("kernels: no backend registered for device kind ", kind,
                     " (register one via RegisterKernelBackend before dispatching)"));
  }
  return backend;
}

}  // namespace

std::string_view ProviderIdName(ProviderId value) noexcept {
  switch (value) {
    case ProviderId::kHpcOps:
      return "hpc_ops";
    case ProviderId::kFlashInfer:
      return "flashinfer";
    case ProviderId::kCutlass:
      return "cutlass";
    case ProviderId::kCublasLt:
      return "cublaslt";
    case ProviderId::kInferxOwned:
      return "inferx_owned";
  }
  return "unknown";
}

std::span<const ProviderId> ProviderPreferenceChain() noexcept {
  return std::span<const ProviderId>(kPreferenceChain);
}

absl::Status RegisterKernelBackend(std::unique_ptr<KernelBackend> backend) {
  if (backend == nullptr) {
    return absl::InvalidArgumentError("kernels: backend registration requires a backend");
  }
  const auto kind = static_cast<uint8_t>(backend->kind());
  if (kind >= kBackendSlotCount) {
    return absl::InvalidArgumentError("kernels: backend device kind out of range");
  }
  if (Backends()[kind] != nullptr) {
    return absl::AlreadyExistsError(
        absl::StrCat("kernels: backend already registered for device kind ", kind));
  }
  Backends()[kind] = std::move(backend);
  return absl::OkStatus();
}

void ResetKernelBackendsForTest() noexcept {
  for (auto& slot : Backends()) slot = nullptr;
}

KernelBackend* FindKernelBackend(DeviceKind kind) noexcept {
  const auto index = static_cast<uint8_t>(kind);
  if (index >= kBackendSlotCount) {
    return nullptr;
  }
  return Backends()[index].get();
}

absl::Status LaunchEmbedding(const ops::EmbeddingRequest& request,
                             const KernelExecutionContext& context,
                             std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) {
    return backend.status();
  }
  return (*backend)->LaunchEmbedding(request, context, forced);
}

absl::Status LaunchGemm(const ops::GemmRequest& request, const KernelExecutionContext& context,
                        std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) {
    return backend.status();
  }
  return (*backend)->LaunchGemm(request, context, forced);
}

absl::Status LaunchLogits(const ops::LogitsRequest& request, const KernelExecutionContext& context,
                          std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) {
    return backend.status();
  }
  return (*backend)->LaunchLogits(request, context, forced);
}

absl::Status LaunchRmsNorm(const ops::RmsNormRequest& request,
                           const KernelExecutionContext& context,
                           std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) {
    return backend.status();
  }
  return (*backend)->LaunchRmsNorm(request, context, forced);
}

absl::Status LaunchRope(const ops::RopeRequest& request, const KernelExecutionContext& context,
                        std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) {
    return backend.status();
  }
  return (*backend)->LaunchRope(request, context, forced);
}

absl::Status LaunchSwiGlu(const ops::SwiGluRequest& request, const KernelExecutionContext& context,
                          std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) {
    return backend.status();
  }
  return (*backend)->LaunchSwiGlu(request, context, forced);
}

absl::Status LaunchResidual(const ops::ResidualRequest& request,
                            const KernelExecutionContext& context,
                            std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) {
    return backend.status();
  }
  return (*backend)->LaunchResidual(request, context, forced);
}

absl::Status LaunchAttention(const ops::AttentionRequest& request,
                             const DeviceAttentionMetadata& metadata,
                             const KernelExecutionContext& context,
                             std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) {
    return backend.status();
  }
  return (*backend)->LaunchAttention(request, metadata, context, forced);
}

absl::Status LaunchActMul(const ActMulRequest& request, const KernelExecutionContext& context,
                          std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchActMul(request, context, forced);
}
absl::Status LaunchAdd3(const Add3Request& request, const KernelExecutionContext& context,
                        std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchAdd3(request, context, forced);
}
absl::Status LaunchFusedAddRmsNorm(const FusedAddRmsNormRequest& request,
                                   const KernelExecutionContext& context,
                                   std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchFusedAddRmsNorm(request, context, forced);
}
absl::Status LaunchGemmaRmsNorm(const GemmaRmsNormRequest& request,
                                const KernelExecutionContext& context,
                                std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchGemmaRmsNorm(request, context, forced);
}
absl::Status LaunchQkRmsNorm(const QkRmsNormRequest& request, const KernelExecutionContext& context,
                             std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchQkRmsNorm(request, context, forced);
}
absl::Status LaunchHadamardTransform(const HadamardTransformRequest& request,
                                     const KernelExecutionContext& context,
                                     std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchHadamardTransform(request, context, forced);
}
absl::Status LaunchArgmax(const ArgmaxRequest& request, const KernelExecutionContext& context,
                          std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchArgmax(request, context, forced);
}
absl::Status LaunchTopPRenorm(const TopPRenormRequest& request,
                              const KernelExecutionContext& context,
                              std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchTopPRenorm(request, context, forced);
}
absl::Status LaunchTopKRenorm(const TopKRenormRequest& request,
                              const KernelExecutionContext& context,
                              std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchTopKRenorm(request, context, forced);
}
absl::Status LaunchFp8Quant(const Fp8QuantRequest& request, const KernelExecutionContext& context,
                            std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchFp8Quant(request, context, forced);
}
absl::Status LaunchSoftmaxTopK(const SoftmaxTopKRequest& request,
                               const KernelExecutionContext& context,
                               std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchSoftmaxTopK(request, context, forced);
}
absl::Status LaunchSigmoidBiasTopK(const SigmoidBiasTopKRequest& request,
                                   const KernelExecutionContext& context,
                                   std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchSigmoidBiasTopK(request, context, forced);
}
absl::Status LaunchAttnRes(const AttnResRequest& request, const KernelExecutionContext& context,
                           std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchAttnRes(request, context, forced);
}
absl::Status LaunchHcMix(const HcMixRequest& request, const KernelExecutionContext& context,
                         std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchHcMix(request, context, forced);
}
absl::Status LaunchHcCombine(const HcCombineRequest& request, const KernelExecutionContext& context,
                             std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchHcCombine(request, context, forced);
}
absl::Status LaunchMhcPre(const MhcPreRequest& request, const KernelExecutionContext& context,
                          std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchMhcPre(request, context, forced);
}
absl::Status LaunchMhcPost(const MhcPostRequest& request, const KernelExecutionContext& context,
                           std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchMhcPost(request, context, forced);
}
absl::Status LaunchRingSconv(const RingSconvRequest& request, const KernelExecutionContext& context,
                             std::optional<ProviderId> forced) {
  absl::StatusOr<KernelBackend*> backend = BackendFor(context);
  if (!backend.ok()) return backend.status();
  return (*backend)->LaunchRingSconv(request, context, forced);
}

}  // namespace inferx::kernels
