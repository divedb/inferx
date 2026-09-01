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
  [[nodiscard]] absl::Status LaunchAttention(
      const ops::AttentionRequest& request, const DeviceAttentionMetadata& metadata,
      const KernelExecutionContext& context, std::optional<ProviderId> forced) override;

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
};

// Assembles the CUDA backend with every provider compiled into this build
// (owned kernels always; CUTLASS/FlashInfer/hpc-ops when their options and
// gitlinks are enabled) and registers it with the neutral dispatcher.
// Per-launch compute capability comes from KernelExecutionContext.
[[nodiscard]] absl::Status RegisterCudaKernelBackend();

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_DISPATCH_CUDA_KERNEL_BACKEND_H_
