// Unified, backend-neutral kernel dispatch surface (ADR 0031).
//
// Upper layers launch operators through the free functions declared in
// inferx/kernels/ops/<category>.h. Dispatch resolves in two steps at run
// time: (1) the DeviceKind of KernelExecutionContext::device selects the
// registered backend, (2) the backend walks the provider preference chain
// for the operator unless the caller forces a provider.
//
// This header and everything under kernels/include must stay free of CUDA
// (and any other backend) types; backends live under kernels/<backend>.
#ifndef INFERX_KERNELS_KERNEL_DISPATCH_H_
#define INFERX_KERNELS_KERNEL_DISPATCH_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/kernels/provider.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/embedding.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/logits.h"
#include "inferx/ops/op_context.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/ops/rope.h"
#include "inferx/ops/swiglu.h"
#include "inferx/tensor/device.h"

namespace inferx::kernels {

// Kernels-layer contracts (defined in inferx/kernels/ops/*.h; forward
// declarations here keep this header free of include cycles).
struct ActMulRequest;
struct Add3Request;
struct FusedAddRmsNormRequest;
struct GemmaRmsNormRequest;
struct QkRmsNormRequest;
struct HadamardTransformRequest;
struct ArgmaxRequest;
struct TopPRenormRequest;
struct TopKRenormRequest;
struct Fp8QuantRequest;
struct SoftmaxTopKRequest;
struct SigmoidBiasTopKRequest;
struct AttnResRequest;
struct HcMixRequest;
struct HcCombineRequest;
struct MhcPreRequest;
struct MhcPostRequest;
struct RingSconvRequest;

// Optional sink for launch failures. The CUDA backend implements sticky-fault
// classification internally and forwards classified failures so device health
// poisoning can stay above the kernels layer.
class KernelFailureObserver {
 public:
  virtual ~KernelFailureObserver() = default;
  virtual void OnKernelFailure(std::string_view operation, bool sticky) = 0;
};

// Backend-neutral launch context. `stream_handle` is an opaque backend stream
// (cudaStream_t for the CUDA backend, reinterpreted by the backend only).
// `compute_capability` is major*10+minor (e.g. 89) or 0 when unknown.
// `workspace` is caller-owned device scratch: providers that need launch
// metadata (e.g. paged-attention page tables) marshal into it and reject
// with ResourceExhausted when `workspace_bytes` is insufficient.
struct KernelExecutionContext {
  Device device = Device::Host();
  uint64_t stream_handle = 0;
  uint16_t compute_capability = 0;
  KernelFailureObserver* observer = nullptr;
  void* workspace = nullptr;
  uint64_t workspace_bytes = 0;
};

// Device-side attention metadata: addresses of the indptr/positions/
// kv-lengths arrays (typically produced by the platform metadata ring). They
// must remain alive through the caller's completion fence.
struct DeviceAttentionMetadata {
  const int32_t* query_indptr = nullptr;
  const int32_t* new_kv_indptr = nullptr;
  const int32_t* query_positions = nullptr;
  const int32_t* kv_lengths_before = nullptr;
};

// A backend implements execution for one DeviceKind. Per-op Launch methods
// default to Unimplemented; a backend overrides what it supports, so adding
// kernels/amd or kernels/npu later touches no upper-layer code.
class KernelBackend {
 public:
  virtual ~KernelBackend() = default;

  [[nodiscard]] virtual DeviceKind kind() const noexcept = 0;

  // Resolves the provider that would serve `kind` on hardware with
  // `compute_capability`. Forced providers that are unavailable return
  // Unimplemented rather than falling through the chain.
  [[nodiscard]] virtual absl::StatusOr<ProviderId> SelectProvider(
      ops::OpKind kind, uint16_t compute_capability, std::optional<ProviderId> forced) const = 0;

  [[nodiscard]] virtual absl::Status LaunchEmbedding(const ops::EmbeddingRequest& /*request*/,
                                                     const KernelExecutionContext&,
                                                     std::optional<ProviderId> /*forced*/) {
    return absl::UnimplementedError("kernels: backend does not implement embedding");
  }
  [[nodiscard]] virtual absl::Status LaunchGemm(const ops::GemmRequest&,
                                                const KernelExecutionContext&,
                                                std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement gemm");
  }
  [[nodiscard]] virtual absl::Status LaunchLogits(const ops::LogitsRequest&,
                                                  const KernelExecutionContext&,
                                                  std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement logits");
  }
  [[nodiscard]] virtual absl::Status LaunchRmsNorm(const ops::RmsNormRequest&,
                                                   const KernelExecutionContext&,
                                                   std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement rms_norm");
  }
  [[nodiscard]] virtual absl::Status LaunchRope(const ops::RopeRequest&,
                                                const KernelExecutionContext&,
                                                std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement rope");
  }
  [[nodiscard]] virtual absl::Status LaunchSwiGlu(const ops::SwiGluRequest&,
                                                  const KernelExecutionContext&,
                                                  std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement silu_multiply");
  }
  [[nodiscard]] virtual absl::Status LaunchResidual(const ops::ResidualRequest&,
                                                    const KernelExecutionContext&,
                                                    std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement residual");
  }
  [[nodiscard]] virtual absl::Status LaunchAttention(const ops::AttentionRequest&,
                                                     const DeviceAttentionMetadata&,
                                                     const KernelExecutionContext&,
                                                     std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement attention");
  }
  [[nodiscard]] virtual absl::Status LaunchActMul(const ActMulRequest&,
                                                  const KernelExecutionContext&,
                                                  std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement act_mul");
  }
  [[nodiscard]] virtual absl::Status LaunchAdd3(const Add3Request&, const KernelExecutionContext&,
                                                std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement add3");
  }
  [[nodiscard]] virtual absl::Status LaunchFusedAddRmsNorm(const FusedAddRmsNormRequest&,
                                                           const KernelExecutionContext&,
                                                           std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement fused_add_rms_norm");
  }
  [[nodiscard]] virtual absl::Status LaunchGemmaRmsNorm(const GemmaRmsNormRequest&,
                                                        const KernelExecutionContext&,
                                                        std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement gemma_rms_norm");
  }
  [[nodiscard]] virtual absl::Status LaunchQkRmsNorm(const QkRmsNormRequest&,
                                                     const KernelExecutionContext&,
                                                     std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement qk_rms_norm");
  }
  [[nodiscard]] virtual absl::Status LaunchHadamardTransform(const HadamardTransformRequest&,
                                                             const KernelExecutionContext&,
                                                             std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement hadamard_transform");
  }
  [[nodiscard]] virtual absl::Status LaunchArgmax(const ArgmaxRequest&,
                                                  const KernelExecutionContext&,
                                                  std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement argmax");
  }
  [[nodiscard]] virtual absl::Status LaunchTopPRenorm(const TopPRenormRequest&,
                                                      const KernelExecutionContext&,
                                                      std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement top_p_renorm");
  }
  [[nodiscard]] virtual absl::Status LaunchTopKRenorm(const TopKRenormRequest&,
                                                      const KernelExecutionContext&,
                                                      std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement top_k_renorm");
  }
  [[nodiscard]] virtual absl::Status LaunchFp8Quant(const Fp8QuantRequest&,
                                                    const KernelExecutionContext&,
                                                    std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement fp8_quant");
  }
  [[nodiscard]] virtual absl::Status LaunchSoftmaxTopK(const SoftmaxTopKRequest&,
                                                       const KernelExecutionContext&,
                                                       std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement softmax_top_k");
  }
  [[nodiscard]] virtual absl::Status LaunchSigmoidBiasTopK(const SigmoidBiasTopKRequest&,
                                                           const KernelExecutionContext&,
                                                           std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement sigmoid_bias_top_k");
  }
  [[nodiscard]] virtual absl::Status LaunchAttnRes(const AttnResRequest&,
                                                   const KernelExecutionContext&,
                                                   std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement attn_res");
  }
  [[nodiscard]] virtual absl::Status LaunchHcMix(const HcMixRequest&, const KernelExecutionContext&,
                                                 std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement hc_mix");
  }
  [[nodiscard]] virtual absl::Status LaunchHcCombine(const HcCombineRequest&,
                                                     const KernelExecutionContext&,
                                                     std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement hc_combine");
  }
  [[nodiscard]] virtual absl::Status LaunchMhcPre(const MhcPreRequest&,
                                                  const KernelExecutionContext&,
                                                  std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement mhc_pre");
  }
  [[nodiscard]] virtual absl::Status LaunchMhcPost(const MhcPostRequest&,
                                                   const KernelExecutionContext&,
                                                   std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement mhc_post");
  }
  [[nodiscard]] virtual absl::Status LaunchRingSconv(const RingSconvRequest&,
                                                     const KernelExecutionContext&,
                                                     std::optional<ProviderId>) {
    return absl::UnimplementedError("kernels: backend does not implement ring_sconv");
  }
};

[[nodiscard]] absl::Status LaunchActMul(const ActMulRequest& request,
                                        const KernelExecutionContext& context,
                                        std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchAdd3(const Add3Request& request,
                                      const KernelExecutionContext& context,
                                      std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchFusedAddRmsNorm(const FusedAddRmsNormRequest& request,
                                                 const KernelExecutionContext& context,
                                                 std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchGemmaRmsNorm(const GemmaRmsNormRequest& request,
                                              const KernelExecutionContext& context,
                                              std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchQkRmsNorm(const QkRmsNormRequest& request,
                                           const KernelExecutionContext& context,
                                           std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchHadamardTransform(const HadamardTransformRequest& request,
                                                   const KernelExecutionContext& context,
                                                   std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchArgmax(const ArgmaxRequest& request,
                                        const KernelExecutionContext& context,
                                        std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchTopPRenorm(const TopPRenormRequest& request,
                                            const KernelExecutionContext& context,
                                            std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchTopKRenorm(const TopKRenormRequest& request,
                                            const KernelExecutionContext& context,
                                            std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchFp8Quant(const Fp8QuantRequest& request,
                                          const KernelExecutionContext& context,
                                          std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchSoftmaxTopK(const SoftmaxTopKRequest& request,
                                             const KernelExecutionContext& context,
                                             std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchSigmoidBiasTopK(const SigmoidBiasTopKRequest& request,
                                                 const KernelExecutionContext& context,
                                                 std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchAttnRes(const AttnResRequest& request,
                                         const KernelExecutionContext& context,
                                         std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchHcMix(const HcMixRequest& request,
                                       const KernelExecutionContext& context,
                                       std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchHcCombine(const HcCombineRequest& request,
                                           const KernelExecutionContext& context,
                                           std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchMhcPre(const MhcPreRequest& request,
                                        const KernelExecutionContext& context,
                                        std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchMhcPost(const MhcPostRequest& request,
                                         const KernelExecutionContext& context,
                                         std::optional<ProviderId> forced = std::nullopt);
[[nodiscard]] absl::Status LaunchRingSconv(const RingSconvRequest& request,
                                           const KernelExecutionContext& context,
                                           std::optional<ProviderId> forced = std::nullopt);

// Registers the backend for its DeviceKind; one backend per kind. The
// registration is explicit (no static initialization) and process-wide.
[[nodiscard]] absl::Status RegisterKernelBackend(std::unique_ptr<KernelBackend> backend);

// Test/teardown hook: drops all registered backends.
void ResetKernelBackendsForTest() noexcept;

// Returns the registered backend for `kind`, or nullptr when absent.
[[nodiscard]] KernelBackend* FindKernelBackend(DeviceKind kind) noexcept;

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_KERNEL_DISPATCH_H_
