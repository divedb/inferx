// CUDA backend for the unified kernels dispatcher (ADR 0031).
//
// Providers implement one operator each and register with the backend; the
// backend resolves launches through the fixed preference chain
// (hpc-ops -> flashinfer -> cutlass -> vendor/owned) unless the caller forces
// a provider, in which case an unavailable provider is an Unimplemented
// error rather than a silent fallback.
#ifndef INFERX_KERNELS_CUDA_DISPATCH_CUDA_KERNEL_BACKEND_H_
#define INFERX_KERNELS_CUDA_DISPATCH_CUDA_KERNEL_BACKEND_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cuda_launch_context.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/ops/activation.h"
#include "inferx/kernels/ops/layernorm.h"
#include "inferx/kernels/ops/model_fused.h"
#include "inferx/kernels/ops/sampling.h"
#include "inferx/kernels/ops/transform.h"
#include "inferx/kernels/provider.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/embedding.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/logits.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/ops/rope.h"
#include "inferx/ops/swiglu.h"

namespace inferx::kernels::cuda {

class EmbeddingProvider {
 public:
  virtual ~EmbeddingProvider() = default;
  // Availability is a function of the request (dtype/shape envelope) and the
  // target compute capability; the chain consults it before every launch.
  [[nodiscard]] virtual ProviderId provider_id() const noexcept = 0;
  [[nodiscard]] virtual bool Available(const ops::EmbeddingRequest* probe,
                                       uint16_t compute_capability) const noexcept = 0;
  [[nodiscard]] virtual absl::Status Launch(const ops::EmbeddingRequest& request,
                                            const CudaLaunchContext& context) const = 0;
};

class GemmProvider {
 public:
  virtual ~GemmProvider() = default;
  [[nodiscard]] virtual ProviderId provider_id() const noexcept = 0;
  [[nodiscard]] virtual bool Available(const ops::GemmRequest* probe,
                                       uint16_t compute_capability) const noexcept = 0;
  [[nodiscard]] virtual absl::Status Launch(const ops::GemmRequest& request,
                                            const CudaLaunchContext& context) const = 0;
};

class LogitsProvider {
 public:
  virtual ~LogitsProvider() = default;
  [[nodiscard]] virtual ProviderId provider_id() const noexcept = 0;
  [[nodiscard]] virtual bool Available(const ops::LogitsRequest* probe,
                                       uint16_t compute_capability) const noexcept = 0;
  [[nodiscard]] virtual absl::Status Launch(const ops::LogitsRequest& request,
                                            const CudaLaunchContext& context) const = 0;
};

class RmsNormProvider {
 public:
  virtual ~RmsNormProvider() = default;
  [[nodiscard]] virtual ProviderId provider_id() const noexcept = 0;
  [[nodiscard]] virtual bool Available(const ops::RmsNormRequest* probe,
                                       uint16_t compute_capability) const noexcept = 0;
  [[nodiscard]] virtual absl::Status Launch(const ops::RmsNormRequest& request,
                                            const CudaLaunchContext& context) const = 0;
};

class RopeProvider {
 public:
  virtual ~RopeProvider() = default;
  [[nodiscard]] virtual ProviderId provider_id() const noexcept = 0;
  [[nodiscard]] virtual bool Available(const ops::RopeRequest* probe,
                                       uint16_t compute_capability) const noexcept = 0;
  [[nodiscard]] virtual absl::Status Launch(const ops::RopeRequest& request,
                                            const CudaLaunchContext& context) const = 0;
};

class SwiGluProvider {
 public:
  virtual ~SwiGluProvider() = default;
  [[nodiscard]] virtual ProviderId provider_id() const noexcept = 0;
  [[nodiscard]] virtual bool Available(const ops::SwiGluRequest* probe,
                                       uint16_t compute_capability) const noexcept = 0;
  [[nodiscard]] virtual absl::Status Launch(const ops::SwiGluRequest& request,
                                            const CudaLaunchContext& context) const = 0;
};

class ResidualProvider {
 public:
  virtual ~ResidualProvider() = default;
  [[nodiscard]] virtual ProviderId provider_id() const noexcept = 0;
  [[nodiscard]] virtual bool Available(const ops::ResidualRequest* probe,
                                       uint16_t compute_capability) const noexcept = 0;
  [[nodiscard]] virtual absl::Status Launch(const ops::ResidualRequest& request,
                                            const CudaLaunchContext& context) const = 0;
};

class AttentionProvider {
 public:
  virtual ~AttentionProvider() = default;
  [[nodiscard]] virtual ProviderId provider_id() const noexcept = 0;
  [[nodiscard]] virtual bool Available(const ops::AttentionRequest* probe,
                                       uint16_t compute_capability) const noexcept = 0;
  // `metadata` carries device addresses of the indptr/positions/kv-lengths
  // arrays; they must outlive the caller's completion fence.
  [[nodiscard]] virtual absl::Status Launch(const ops::AttentionRequest& request,
                                            const DeviceAttentionMetadata& metadata,
                                            const CudaLaunchContext& context) const = 0;
};

// Generic per-op provider for kernels-layer contracts (ADR 0031 taxonomy):
// availability is a function of the request probe and compute capability.
template <typename Request>
class OpProvider {
 public:
  virtual ~OpProvider() = default;
  [[nodiscard]] virtual ProviderId provider_id() const noexcept = 0;
  [[nodiscard]] virtual bool Available(const Request* probe,
                                       uint16_t compute_capability) const noexcept = 0;
  [[nodiscard]] virtual absl::Status Launch(const Request& request,
                                            const CudaLaunchContext& context) const = 0;
};

using ActMulProvider = OpProvider<ActMulRequest>;
using Add3Provider = OpProvider<Add3Request>;
using FusedAddRmsNormProvider = OpProvider<FusedAddRmsNormRequest>;
using GemmaRmsNormProvider = OpProvider<GemmaRmsNormRequest>;
using QkRmsNormProvider = OpProvider<QkRmsNormRequest>;
using HadamardTransformProvider = OpProvider<HadamardTransformRequest>;
using ArgmaxProvider = OpProvider<ArgmaxRequest>;
using TopPRenormProvider = OpProvider<TopPRenormRequest>;
using TopKRenormProvider = OpProvider<TopKRenormRequest>;
using Fp8QuantProvider = OpProvider<Fp8QuantRequest>;
using SoftmaxTopKProvider = OpProvider<SoftmaxTopKRequest>;
using SigmoidBiasTopKProvider = OpProvider<SigmoidBiasTopKRequest>;
using AttnResProvider = OpProvider<AttnResRequest>;
using HcMixProvider = OpProvider<HcMixRequest>;
using HcCombineProvider = OpProvider<HcCombineRequest>;
using MhcPreProvider = OpProvider<MhcPreRequest>;
using MhcPostProvider = OpProvider<MhcPostRequest>;
using RingSconvProvider = OpProvider<RingSconvRequest>;

class CudaKernelBackend final : public KernelBackend {
 public:
  void AddEmbeddingProvider(std::unique_ptr<EmbeddingProvider> provider);
  void AddGemmProvider(std::unique_ptr<GemmProvider> provider);
  void AddLogitsProvider(std::unique_ptr<LogitsProvider> provider);
  void AddRmsNormProvider(std::unique_ptr<RmsNormProvider> provider);
  void AddRopeProvider(std::unique_ptr<RopeProvider> provider);
  void AddSwiGluProvider(std::unique_ptr<SwiGluProvider> provider);
  void AddResidualProvider(std::unique_ptr<ResidualProvider> provider);
  void AddAttentionProvider(std::unique_ptr<AttentionProvider> provider);
  void AddActMulProvider(std::unique_ptr<ActMulProvider> provider);
  void AddAdd3Provider(std::unique_ptr<Add3Provider> provider);
  void AddFusedAddRmsNormProvider(std::unique_ptr<FusedAddRmsNormProvider> provider);
  void AddGemmaRmsNormProvider(std::unique_ptr<GemmaRmsNormProvider> provider);
  void AddQkRmsNormProvider(std::unique_ptr<QkRmsNormProvider> provider);
  void AddHadamardTransformProvider(std::unique_ptr<HadamardTransformProvider> provider);
  void AddArgmaxProvider(std::unique_ptr<ArgmaxProvider> provider);
  void AddTopPRenormProvider(std::unique_ptr<TopPRenormProvider> provider);
  void AddTopKRenormProvider(std::unique_ptr<TopKRenormProvider> provider);
  void AddFp8QuantProvider(std::unique_ptr<Fp8QuantProvider> provider);
  void AddSoftmaxTopKProvider(std::unique_ptr<SoftmaxTopKProvider> provider);
  void AddSigmoidBiasTopKProvider(std::unique_ptr<SigmoidBiasTopKProvider> provider);
  void AddAttnResProvider(std::unique_ptr<AttnResProvider> provider);
  void AddHcMixProvider(std::unique_ptr<HcMixProvider> provider);
  void AddHcCombineProvider(std::unique_ptr<HcCombineProvider> provider);
  void AddMhcPreProvider(std::unique_ptr<MhcPreProvider> provider);
  void AddMhcPostProvider(std::unique_ptr<MhcPostProvider> provider);
  void AddRingSconvProvider(std::unique_ptr<RingSconvProvider> provider);

  [[nodiscard]] DeviceKind kind() const noexcept override { return DeviceKind::kCuda; }
  [[nodiscard]] absl::StatusOr<ProviderId> SelectProvider(
      ops::OpKind kind, uint16_t compute_capability,
      std::optional<ProviderId> forced) const override;

  [[nodiscard]] absl::Status LaunchEmbedding(const ops::EmbeddingRequest& request,
                                             const KernelExecutionContext& context,
                                             std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchGemm(const ops::GemmRequest& request,
                                        const KernelExecutionContext& context,
                                        std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchLogits(const ops::LogitsRequest& request,
                                          const KernelExecutionContext& context,
                                          std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchRmsNorm(const ops::RmsNormRequest& request,
                                           const KernelExecutionContext& context,
                                           std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchRope(const ops::RopeRequest& request,
                                        const KernelExecutionContext& context,
                                        std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchSwiGlu(const ops::SwiGluRequest& request,
                                          const KernelExecutionContext& context,
                                          std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchResidual(const ops::ResidualRequest& request,
                                            const KernelExecutionContext& context,
                                            std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchAttention(const ops::AttentionRequest& request,
                                             const DeviceAttentionMetadata& metadata,
                                             const KernelExecutionContext& context,
                                             std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchActMul(const ActMulRequest& request,
                                          const KernelExecutionContext& context,
                                          std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchAdd3(const Add3Request& request,
                                        const KernelExecutionContext& context,
                                        std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchFusedAddRmsNorm(const FusedAddRmsNormRequest& request,
                                                   const KernelExecutionContext& context,
                                                   std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchGemmaRmsNorm(const GemmaRmsNormRequest& request,
                                                const KernelExecutionContext& context,
                                                std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchQkRmsNorm(const QkRmsNormRequest& request,
                                             const KernelExecutionContext& context,
                                             std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchHadamardTransform(const HadamardTransformRequest& request,
                                                     const KernelExecutionContext& context,
                                                     std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchArgmax(const ArgmaxRequest& request,
                                          const KernelExecutionContext& context,
                                          std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchTopPRenorm(const TopPRenormRequest& request,
                                              const KernelExecutionContext& context,
                                              std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchTopKRenorm(const TopKRenormRequest& request,
                                              const KernelExecutionContext& context,
                                              std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchFp8Quant(const Fp8QuantRequest& request,
                                            const KernelExecutionContext& context,
                                            std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchSoftmaxTopK(const SoftmaxTopKRequest& request,
                                               const KernelExecutionContext& context,
                                               std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchSigmoidBiasTopK(const SigmoidBiasTopKRequest& request,
                                                   const KernelExecutionContext& context,
                                                   std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchAttnRes(const AttnResRequest& request,
                                           const KernelExecutionContext& context,
                                           std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchHcMix(const HcMixRequest& request,
                                         const KernelExecutionContext& context,
                                         std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchHcCombine(const HcCombineRequest& request,
                                             const KernelExecutionContext& context,
                                             std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchMhcPre(const MhcPreRequest& request,
                                          const KernelExecutionContext& context,
                                          std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchMhcPost(const MhcPostRequest& request,
                                           const KernelExecutionContext& context,
                                           std::optional<ProviderId> forced) override;
  [[nodiscard]] absl::Status LaunchRingSconv(const RingSconvRequest& request,
                                             const KernelExecutionContext& context,
                                             std::optional<ProviderId> forced) override;

 private:
  template <typename Provider, typename Request>
  absl::StatusOr<Provider*> Resolve(const std::vector<std::unique_ptr<Provider>>& providers,
                                    const Request* probe, uint16_t compute_capability,
                                    std::optional<ProviderId> forced, const char* op_name) const;

  std::vector<std::unique_ptr<EmbeddingProvider>> embedding_;
  std::vector<std::unique_ptr<GemmProvider>> gemm_;
  std::vector<std::unique_ptr<LogitsProvider>> logits_;
  std::vector<std::unique_ptr<RmsNormProvider>> rms_norm_;
  std::vector<std::unique_ptr<RopeProvider>> rope_;
  std::vector<std::unique_ptr<SwiGluProvider>> swiglu_;
  std::vector<std::unique_ptr<ResidualProvider>> residual_;
  std::vector<std::unique_ptr<AttentionProvider>> attention_;
  std::vector<std::unique_ptr<ActMulProvider>> act_mul_;
  std::vector<std::unique_ptr<Add3Provider>> add3_;
  std::vector<std::unique_ptr<FusedAddRmsNormProvider>> fused_add_rms_norm_;
  std::vector<std::unique_ptr<GemmaRmsNormProvider>> gemma_rms_norm_;
  std::vector<std::unique_ptr<QkRmsNormProvider>> qk_rms_norm_;
  std::vector<std::unique_ptr<HadamardTransformProvider>> hadamard_;
  std::vector<std::unique_ptr<ArgmaxProvider>> argmax_;
  std::vector<std::unique_ptr<TopPRenormProvider>> top_p_renorm_;
  std::vector<std::unique_ptr<TopKRenormProvider>> top_k_renorm_;
  std::vector<std::unique_ptr<Fp8QuantProvider>> fp8_quant_;
  std::vector<std::unique_ptr<SoftmaxTopKProvider>> softmax_topk_;
  std::vector<std::unique_ptr<SigmoidBiasTopKProvider>> sigmoid_bias_topk_;
  std::vector<std::unique_ptr<AttnResProvider>> attn_res_;
  std::vector<std::unique_ptr<HcMixProvider>> hc_mix_;
  std::vector<std::unique_ptr<HcCombineProvider>> hc_combine_;
  std::vector<std::unique_ptr<MhcPreProvider>> mhc_pre_;
  std::vector<std::unique_ptr<MhcPostProvider>> mhc_post_;
  std::vector<std::unique_ptr<RingSconvProvider>> ring_sconv_;
};

// Assembles the CUDA backend with every provider compiled into this build
// (owned kernels always; CUTLASS/FlashInfer/hpc-ops when their options and
// gitlinks are enabled) and registers it with the neutral dispatcher.
// Per-launch compute capability comes from KernelExecutionContext.
[[nodiscard]] absl::Status RegisterCudaKernelBackend();

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_DISPATCH_CUDA_KERNEL_BACKEND_H_
