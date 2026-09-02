#include "op_launchers.h"

#include <cstdint>
#include <vector>

#include "cuda_tensor_checks.h"
#include "inferx/kernels/cuda/activation_kernels.h"
#include "inferx/kernels/cuda/attention_kernels.h"
#include "inferx/kernels/cuda/gemm_kernels.h"
#include "inferx/kernels/cuda/layernorm_kernels.h"
#include "inferx/kernels/cuda/model_fused_kernels.h"
#include "inferx/kernels/cuda/quantization_kernels.h"
#include "inferx/kernels/cuda/sampling_kernels.h"
#include "inferx/kernels/cuda/transform_kernels.h"

namespace inferx::kernels::cuda::launchers {
namespace {

constexpr uint32_t kActMulSilu = 0;

// Chain probes are null; every op serves any CUDA device (>= sm80 for the
// flashinfer kernels), with request-envelope checks applied per probe.
bool AlwaysAvailable(const void*, uint16_t compute_capability) {
  return compute_capability == 0 || compute_capability >= 80;
}

absl::Status RequireDevice(const TensorView& tensor, const CudaLaunchContext& context,
                           const char* field) {
  if (tensor.buffer().device().kind != DeviceKind::kCuda ||
      tensor.buffer().memory_kind() != MemoryKind::kDevice) {
    return absl::InvalidArgumentError(std::string(field) + ": CUDA device memory is required");
  }
  return ValidateDeviceMatch(tensor, context.device);
}

}  // namespace

absl::Status ActMul(const ActMulRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateActMul(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.input, context, "kernels_cuda.act_mul");
  if (!status.ok()) return status;
  const uint32_t kind = request.kind == ActMulKind::kSilu
                            ? kActMulSilu
                            : (request.kind == ActMulKind::kGelu ? 1U : 2U);
  return CheckLaunchResult(activation::LaunchActMulKernel(
                               Address(request.input), Address(request.output),
                               request.input.shape().dim(0), request.input.shape().dim(1) / 2, kind,
                               ToKernelStorageType(request.input.dtype()), context.stream),
                           "kernels_cuda.act_mul.launch", context);
}

bool ActMulAvailable(const ActMulRequest* probe, uint16_t compute_capability) {
  if (!AlwaysAvailable(nullptr, compute_capability)) return false;
  return probe == nullptr || probe->input.dtype() != Dtype::kUInt8;
}

absl::Status Add3(const Add3Request& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateAdd3(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.a, context, "kernels_cuda.add3");
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> elements = request.a.shape().NumElements();
  if (!elements.ok()) return elements.status();
  return CheckLaunchResult(
      activation::LaunchAdd3Kernel(Address(request.a), Address(request.b), Address(request.c),
                                   Address(request.output), *elements,
                                   ToKernelStorageType(request.a.dtype()), context.stream),
      "kernels_cuda.add3.launch", context);
}

bool Add3Available(const Add3Request* /*probe*/, uint16_t compute_capability) {
  return AlwaysAvailable(nullptr, compute_capability);
}

absl::Status FusedAddRmsNorm(const FusedAddRmsNormRequest& request,
                             const CudaLaunchContext& context) {
  absl::Status status = ValidateFusedAddRmsNorm(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.input.AsConst(), context, "kernels_cuda.fused_add_rmsnorm");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      layernorm::LaunchFlashInferFusedAddRmsNorm(
          Address(request.input), Address(request.residual), Address(request.weight),
          request.input.shape().dim(0), request.input.shape().dim(1), request.epsilon,
          ToKernelStorageType(request.input.dtype()), context.stream),
      "kernels_cuda.fused_add_rmsnorm.launch", context);
}

bool FusedAddRmsNormAvailable(const FusedAddRmsNormRequest* probe, uint16_t compute_capability) {
  if (!AlwaysAvailable(nullptr, compute_capability)) return false;
  if (probe == nullptr) return true;
  const uint32_t vec = probe->input.dtype() == Dtype::kFloat32 ? 4 : 8;
  return probe->input.shape().rank() == 2 && probe->input.shape().dim(1) % vec == 0;
}

absl::Status GemmaRmsNorm(const GemmaRmsNormRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateGemmaRmsNorm(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.input, context, "kernels_cuda.gemma_rmsnorm");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      layernorm::LaunchFlashInferGemmaRmsNorm(
          Address(request.input), Address(request.weight), Address(request.output),
          request.input.shape().dim(0), request.input.shape().dim(1), request.epsilon,
          ToKernelStorageType(request.input.dtype()), context.stream),
      "kernels_cuda.gemma_rmsnorm.launch", context);
}

bool GemmaRmsNormAvailable(const GemmaRmsNormRequest* probe, uint16_t compute_capability) {
  if (!AlwaysAvailable(nullptr, compute_capability)) return false;
  if (probe == nullptr) return true;
  const uint32_t vec = probe->input.dtype() == Dtype::kFloat32 ? 4 : 8;
  return probe->input.shape().rank() == 2 && probe->input.shape().dim(1) % vec == 0;
}

absl::Status QkRmsNorm(const QkRmsNormRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateQkRmsNorm(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.query.AsConst(), context, "kernels_cuda.qk_rmsnorm");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      layernorm::LaunchFlashInferQkRmsNorm(
          Address(request.query), Address(request.key), Address(request.query_weight),
          Address(request.key_weight), request.query.shape().dim(0), request.query.shape().dim(1),
          request.key.shape().dim(1), request.query.shape().dim(2), request.epsilon,
          ToKernelStorageType(request.query.dtype()), context.stream),
      "kernels_cuda.qk_rmsnorm.launch", context);
}

bool QkRmsNormAvailable(const QkRmsNormRequest* probe, uint16_t compute_capability) {
  if (!AlwaysAvailable(nullptr, compute_capability)) return false;
  if (probe == nullptr) return true;
  const uint32_t vec = probe->query.dtype() == Dtype::kFloat32 ? 4 : 8;
  return probe->query.shape().rank() == 3 && probe->query.shape().dim(2) % vec == 0;
}

absl::Status HadamardTransform(const HadamardTransformRequest& request,
                               const CudaLaunchContext& context) {
  absl::Status status = ValidateHadamardTransform(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.input, context, "kernels_cuda.hadamard");
  if (!status.ok()) return status;
  const uint64_t rows = request.input.shape().NumElements().value_or(0) / 128;
  return CheckLaunchResult(transform::LaunchHadamard128(
                               Address(request.input), Address(request.output), rows, request.scale,
                               ToKernelStorageType(request.input.dtype()), context.stream),
                           "kernels_cuda.hadamard.launch", context);
}

bool HadamardTransformAvailable(const HadamardTransformRequest* /*probe*/,
                                uint16_t compute_capability) {
  return AlwaysAvailable(nullptr, compute_capability);
}

absl::Status Argmax(const ArgmaxRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateArgmax(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.logits, context, "kernels_cuda.argmax");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      sampling::LaunchArgmax(Address(request.logits), request.logits.shape().dim(0),
                             request.logits.shape().dim(1),
                             static_cast<int32_t*>(Address(request.output)),
                             ToKernelStorageType(request.logits.dtype()), context.stream),
      "kernels_cuda.argmax.launch", context);
}

bool ArgmaxAvailable(const ArgmaxRequest* /*probe*/, uint16_t compute_capability) {
  return AlwaysAvailable(nullptr, compute_capability);
}

absl::Status TopPRenorm(const TopPRenormRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateTopPRenorm(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.probs.AsConst(), context, "kernels_cuda.top_p_renorm");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      sampling::LaunchFlashInferTopPRenorm(
          Address(request.probs), request.probs.shape().dim(0), request.probs.shape().dim(1),
          request.top_p, ToKernelStorageType(request.probs.dtype()), context.stream),
      "kernels_cuda.top_p_renorm.launch", context);
}

bool TopPRenormAvailable(const TopPRenormRequest* probe, uint16_t compute_capability) {
  if (!AlwaysAvailable(nullptr, compute_capability)) return false;
  return probe == nullptr || probe->probs.dtype() == Dtype::kFloat32;
}

absl::Status TopKRenorm(const TopKRenormRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateTopKRenorm(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.probs.AsConst(), context, "kernels_cuda.top_k_renorm");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      sampling::LaunchTopKRenorm(Address(request.probs), request.probs.shape().dim(0),
                                 request.probs.shape().dim(1), request.top_k,
                                 ToKernelStorageType(request.probs.dtype()), context.stream),
      "kernels_cuda.top_k_renorm.launch", context);
}

bool TopKRenormAvailable(const TopKRenormRequest* /*probe*/, uint16_t compute_capability) {
  return AlwaysAvailable(nullptr, compute_capability);
}

absl::Status Fp8Quant(const Fp8QuantRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateFp8Quant(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.input, context, "kernels_cuda.fp8_quant");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      quant::LaunchFp8Quant(Address(request.input), Address(request.output),
                            static_cast<float*>(Address(request.scales)),
                            request.input.shape().dim(0), request.input.shape().dim(1),
                            static_cast<uint32_t>(request.granularity),
                            static_cast<uint32_t>(request.group_size), context.stream),
      "kernels_cuda.fp8_quant.launch", context);
}

bool Fp8QuantAvailable(const Fp8QuantRequest* /*probe*/, uint16_t compute_capability) {
  return AlwaysAvailable(nullptr, compute_capability);
}

absl::Status SoftmaxTopK(const SoftmaxTopKRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateSoftmaxTopK(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.logits, context, "kernels_cuda.softmax_topk");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      moe::LaunchSoftmaxTopK(Address(request.logits), Address(request.weights),
                             static_cast<int32_t*>(Address(request.ids)),
                             request.logits.shape().dim(0), request.logits.shape().dim(1),
                             static_cast<uint32_t>(request.weights.shape().dim(1)),
                             request.renormalize, context.stream),
      "kernels_cuda.softmax_topk.launch", context);
}

bool SoftmaxTopKAvailable(const SoftmaxTopKRequest* probe, uint16_t compute_capability) {
  if (!AlwaysAvailable(nullptr, compute_capability)) return false;
  return probe == nullptr || probe->logits.shape().dim(1) <= 4096;
}

absl::Status SigmoidBiasTopK(const SigmoidBiasTopKRequest& request,
                             const CudaLaunchContext& context) {
  absl::Status status = ValidateSigmoidBiasTopK(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.logits, context, "kernels_cuda.sigmoid_bias_topk");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      moe::LaunchSigmoidBiasTopK(
          Address(request.logits), Address(request.bias), Address(request.weights),
          static_cast<int32_t*>(Address(request.ids)), request.logits.shape().dim(0),
          request.logits.shape().dim(1), static_cast<uint32_t>(request.weights.shape().dim(1)),
          request.routed_scaling_factor, request.renormalize, context.stream),
      "kernels_cuda.sigmoid_bias_topk.launch", context);
}

bool SigmoidBiasTopKAvailable(const SigmoidBiasTopKRequest* probe, uint16_t compute_capability) {
  if (!AlwaysAvailable(nullptr, compute_capability)) return false;
  return probe == nullptr || probe->logits.shape().dim(1) <= 4096;
}

absl::Status AttnRes(const AttnResRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateAttnRes(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.layer_residual.AsConst(), context, "kernels_cuda.attn_res");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      attnres::LaunchAttnRes(
          Address(request.block_residual), Address(request.layer_residual),
          Address(request.res_weight), Address(request.rms_weight),
          request.out_norm_weight.has_value() ? Address(*request.out_norm_weight) : nullptr,
          request.layer_residual.shape().dim(0), request.layer_residual.shape().dim(1),
          request.block_residual.shape().dim(0), request.epsilon, request.out_norm_epsilon,
          ToKernelStorageType(request.layer_residual.dtype()), context.stream),
      "kernels_cuda.attn_res.launch", context);
}

bool AttnResAvailable(const AttnResRequest* probe, uint16_t compute_capability) {
  if (!AlwaysAvailable(nullptr, compute_capability)) return false;
  return probe == nullptr || probe->block_residual.shape().dim(0) + 1 <= 13;
}

absl::Status HcMix(const HcMixRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateHcMix(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.normalized, context, "kernels_cuda.hc_mix");
  if (!status.ok()) return status;
  const uint64_t tokens = request.normalized.shape().dim(0);
  const uint64_t needed =
      hc::HcMixWorkspaceBytes(tokens, request.lowrank, request.hc_count, request.hidden_size);
  if (context.workspace == nullptr || context.workspace_bytes < needed) {
    return absl::ResourceExhaustedError("kernels_cuda.hc_mix: workspace too small");
  }
  return CheckLaunchResult(
      hc::LaunchHcMix(Address(request.normalized), Address(request.projection_weight),
                      Address(request.up_weight), Address(request.mixed),
                      Address(request.inject_logits), context.workspace, tokens, request.lowrank,
                      request.hc_count, request.hidden_size, request.projection_scale,
                      ToKernelStorageType(request.normalized.dtype()), context.stream),
      "kernels_cuda.hc_mix.launch", context);
}

bool HcMixAvailable(const HcMixRequest* /*probe*/, uint16_t compute_capability) {
  return AlwaysAvailable(nullptr, compute_capability);
}

absl::Status HcCombine(const HcCombineRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateHcCombine(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.block_output, context, "kernels_cuda.hc_combine");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      hc::LaunchHcCombine(Address(request.block_output), Address(request.residual),
                          Address(request.inject_logits), request.block_output.shape().dim(0),
                          request.hc_count, request.hidden_size,
                          ToKernelStorageType(request.block_output.dtype()), context.stream),
      "kernels_cuda.hc_combine.launch", context);
}

bool HcCombineAvailable(const HcCombineRequest* /*probe*/, uint16_t compute_capability) {
  return AlwaysAvailable(nullptr, compute_capability);
}

absl::Status MhcPre(const MhcPreRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateMhcPre(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.residual, context, "kernels_cuda.mhc_pre");
  if (!status.ok()) return status;
  const uint64_t tokens = request.residual.shape().dim(0);
  const uint64_t needed =
      mhc::MhcPreWorkspaceBytes(tokens, static_cast<uint32_t>(request.residual.shape().dim(1)));
  if (context.workspace == nullptr || context.workspace_bytes < needed) {
    return absl::ResourceExhaustedError("kernels_cuda.mhc_pre: workspace too small");
  }
  return CheckLaunchResult(
      mhc::LaunchMhcPre(Address(request.residual), Address(request.fn), Address(request.hc_scale),
                        Address(request.hc_base), context.workspace, Address(request.layer_input),
                        Address(request.post), Address(request.comb), tokens,
                        static_cast<uint32_t>(request.residual.shape().dim(1)),
                        static_cast<uint32_t>(request.residual.shape().dim(2)), request.rms_eps,
                        request.hc_eps, static_cast<uint32_t>(request.sinkhorn_iters),
                        ToKernelStorageType(request.residual.dtype()), context.stream),
      "kernels_cuda.mhc_pre.launch", context);
}

bool MhcPreAvailable(const MhcPreRequest* probe, uint16_t compute_capability) {
  if (!AlwaysAvailable(nullptr, compute_capability)) return false;
  return probe == nullptr || probe->residual.shape().dim(1) <= 8;
}

absl::Status MhcPost(const MhcPostRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateMhcPost(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.hidden_states, context, "kernels_cuda.mhc_post");
  if (!status.ok()) return status;
  return CheckLaunchResult(
      mhc::LaunchMhcPost(Address(request.hidden_states), Address(request.residual),
                         Address(request.post), Address(request.comb),
                         request.hidden_states.shape().dim(0),
                         static_cast<uint32_t>(request.residual.shape().dim(1)),
                         static_cast<uint32_t>(request.residual.shape().dim(2)),
                         ToKernelStorageType(request.hidden_states.dtype()), context.stream),
      "kernels_cuda.mhc_post.launch", context);
}

bool MhcPostAvailable(const MhcPostRequest* /*probe*/, uint16_t compute_capability) {
  return AlwaysAvailable(nullptr, compute_capability);
}

absl::Status RingSconv(const RingSconvRequest& request, const CudaLaunchContext& context) {
  absl::Status status = ValidateRingSconv(request);
  if (!status.ok()) return status;
  status = RequireDevice(request.x, context, "kernels_cuda.ring_sconv");
  if (!status.ok()) return status;
  const uint64_t batch = request.seq_lens.shape().dim(0);
  // seq prefix staged through the workspace (batch + 1 int32 words).
  return CheckLaunchResult(
      conv::LaunchRingSconv(
          Address(request.x), Address(request.weight), Address(request.conv_cache),
          static_cast<const int32_t*>(context.workspace),
          static_cast<const int32_t*>(Address(request.cache_indices)), Address(request.output),
          request.x.shape().dim(0), request.x.shape().dim(1), static_cast<uint32_t>(request.window),
          request.conv_cache.shape().dim(1), batch, ToKernelStorageType(request.x.dtype()),
          context.stream),
      "kernels_cuda.ring_sconv.launch", context);
}

bool RingSconvAvailable(const RingSconvRequest* /*probe*/, uint16_t compute_capability) {
  return AlwaysAvailable(nullptr, compute_capability);
}

}  // namespace inferx::kernels::cuda::launchers
