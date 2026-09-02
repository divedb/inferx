#include "flashinfer_providers.h"

#include <vector>

#include "cuda_tensor_checks.h"
#include "inferx/kernels/cuda/attention_kernels.h"
#include "inferx/kernels/cuda/layernorm_kernels.h"
#include "inferx/kernels/cuda/transform_kernels.h"
#include "op_validation.h"

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
    const uint32_t vec = probe->input.dtype() == Dtype::kFloat32 ? 4 : 8;
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
            Address(request.key_output), request.query.shape().dim(0), request.query.shape().dim(1),
            request.key.shape().dim(1), request.query.shape().dim(2), request.theta,
            ToKernelStorageType(request.query.dtype()), context.stream),
        "kernels_cuda.rope.flashinfer.launch", context);
  }
};

// Real FlashInfer attention (ADR 0032 gap closed): the contiguous InferX
// cache maps onto paged_kv_t (page_size = max_context, one page per
// sequence) and the launch marshals page tables/scheduler arrays into the
// context workspace. head_dim in {64, 128}; no split-KV (4096 envelope).
class FlashInferAttentionProvider final : public AttentionProvider {
 public:
  ProviderId provider_id() const noexcept override { return ProviderId::kFlashInfer; }
  bool Available(const ops::AttentionRequest* probe,
                 uint16_t compute_capability) const noexcept override {
    if (compute_capability != 0 && compute_capability < 80) return false;
    if (probe == nullptr) return true;
    const uint64_t head_dimension = probe->query.shape().dim(2);
    return head_dimension == 64 || head_dimension == 128;
  }
  absl::Status Launch(const ops::AttentionRequest& request, const DeviceAttentionMetadata& metadata,
                      const CudaLaunchContext& context) const override {
    absl::Status status = ValidateAttentionForCuda(request, metadata, context);
    if (!status.ok()) return status;
    const uint64_t batch = request.key_cache.shape().dim(0);
    const uint64_t max_context = request.key_cache.shape().dim(1);
    const uint64_t head_dimension = request.query.shape().dim(2);
    const uint64_t total_queries = request.query.shape().dim(0);
    const auto storage = ToKernelStorageType(request.query.dtype());
    if (request.phase == ops::ExecutionPhase::kDecode) {
      const uint64_t words = attention::AttentionWorkspaceWords(batch, batch);
      if (context.workspace == nullptr || context.workspace_bytes < words * sizeof(int32_t)) {
        return absl::ResourceExhaustedError("kernels_cuda.attention: workspace too small");
      }
      return CheckLaunchResult(
          attention::LaunchFlashInferDecode(
              Address(request.query), Address(request.key_cache.AsConst()),
              Address(request.value_cache.AsConst()), metadata.new_kv_indptr,
              metadata.kv_lengths_before, Address(request.output), context.workspace, words, batch,
              max_context, request.query.shape().dim(1), request.new_key.shape().dim(1),
              head_dimension, storage, context.stream),
          "kernels_cuda.attention.flashinfer.decode", context);
    }
    // Prefill: tile enumeration is host-known from the mirrored spans.
    constexpr uint32_t kCtaTileQ = 64;
    std::vector<int32_t> host_q_indptr(request.query_indptr.begin(), request.query_indptr.end());
    std::vector<int32_t> tile_offsets(batch + 1, 0);
    for (uint64_t b = 0; b < batch; ++b) {
      const int64_t qo_len = host_q_indptr[b + 1] - host_q_indptr[b];
      const int64_t tiles_b = (qo_len + kCtaTileQ - 1) / kCtaTileQ;
      tile_offsets[b + 1] = tile_offsets[b] + static_cast<int32_t>(tiles_b);
    }
    const uint64_t tiles = static_cast<uint64_t>(tile_offsets[batch]);
    const uint64_t words = attention::AttentionWorkspaceWords(batch, tiles);
    if (context.workspace == nullptr || context.workspace_bytes < words * sizeof(int32_t)) {
      return absl::ResourceExhaustedError("kernels_cuda.attention: workspace too small");
    }
    return CheckLaunchResult(
        attention::LaunchFlashInferPrefillStaged(
            Address(request.query), Address(request.key_cache.AsConst()),
            Address(request.value_cache.AsConst()), host_q_indptr.data(), tile_offsets.data(),
            metadata.query_indptr, metadata.new_kv_indptr, metadata.kv_lengths_before,
            Address(request.output), context.workspace, words, batch, total_queries, max_context,
            request.query.shape().dim(1), request.new_key.shape().dim(1), head_dimension, storage,
            context.stream),
        "kernels_cuda.attention.flashinfer.prefill", context);
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
