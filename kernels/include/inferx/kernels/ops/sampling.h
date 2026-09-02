// Sampling and MoE-routing contracts.
#ifndef INFERX_KERNELS_OPS_SAMPLING_CONTRACTS_H_
#define INFERX_KERNELS_OPS_SAMPLING_CONTRACTS_H_

#include <cstdint>

#include "absl/status/status.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::kernels {

enum class QuantGranularity : uint8_t { kTensor, kToken, kTokenGroup };

// logits [rows, vocab] -> argmax [rows] int32; all-NaN rows yield -1, ties
// resolve to the lowest index (TokenSpeed argmax semantics).
struct ArgmaxRequest {
  TensorView logits;
  MutableTensorView output;
};

// probs [rows, vocab] renormalized in place over the top-p mass.
struct TopPRenormRequest {
  MutableTensorView probs;
  float top_p = 0.9F;
};

// probs [rows, vocab] renormalized in place over the top-k entries.
struct TopKRenormRequest {
  MutableTensorView probs;
  uint32_t top_k = 32;
};

// FP8 e4m3 quantization: input [tokens, dim] -> output same-shape FP8 plus
// scale factors (per tensor [1] / per token [tokens] / per group
// [tokens, ceil(dim/group)] in FP32). No external provider at the pins:
// hpc-ops quant kernels are fused activations; the pinned flashinfer
// quantization.cuh carries only pack-bits kernels — custom tier.
struct Fp8QuantRequest {
  TensorView input;
  MutableTensorView output;
  MutableTensorView scales;
  QuantGranularity granularity = QuantGranularity::kToken;
  uint32_t group_size = 128;
};

// MoE routing: router logits [tokens, experts] -> topk weights FP32
// [tokens, k], expert ids int32 [tokens, k] (softmax + top-k + renorm).
struct SoftmaxTopKRequest {
  TensorView logits;
  MutableTensorView weights;
  MutableTensorView ids;
  bool renormalize = true;
};

// MoE routing (Kimi-style): weights = routed_scaling *
// sigmoid(logits + bias) of the selected experts; selection by biased score.
struct SigmoidBiasTopKRequest {
  TensorView logits;
  TensorView bias;
  MutableTensorView weights;
  MutableTensorView ids;
  float routed_scaling_factor = 1.0F;
  bool renormalize = false;
};

[[nodiscard]] absl::Status ValidateArgmax(const ArgmaxRequest& request);
[[nodiscard]] absl::Status ReferenceArgmax(const ArgmaxRequest& request);
[[nodiscard]] absl::Status ValidateTopPRenorm(const TopPRenormRequest& request);
[[nodiscard]] absl::Status ReferenceTopPRenorm(const TopPRenormRequest& request);
[[nodiscard]] absl::Status ValidateTopKRenorm(const TopKRenormRequest& request);
[[nodiscard]] absl::Status ReferenceTopKRenorm(const TopKRenormRequest& request);
[[nodiscard]] absl::Status ValidateFp8Quant(const Fp8QuantRequest& request);
[[nodiscard]] absl::Status ReferenceFp8Quant(const Fp8QuantRequest& request);
[[nodiscard]] absl::Status ValidateSoftmaxTopK(const SoftmaxTopKRequest& request);
[[nodiscard]] absl::Status ReferenceSoftmaxTopK(const SoftmaxTopKRequest& request);
[[nodiscard]] absl::Status ValidateSigmoidBiasTopK(const SigmoidBiasTopKRequest& request);
[[nodiscard]] absl::Status ReferenceSigmoidBiasTopK(const SigmoidBiasTopKRequest& request);

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_OPS_SAMPLING_CONTRACTS_H_
