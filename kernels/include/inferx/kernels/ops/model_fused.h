// Model-specific fused residual-stream contracts (TokenSpeed reference
// semantics; no external provider at the pins — custom tier).
#ifndef INFERX_KERNELS_OPS_MODEL_FUSED_CONTRACTS_H_
#define INFERX_KERNELS_OPS_MODEL_FUSED_CONTRACTS_H_

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::kernels {

// Kimi-style attention-output residual mixing (TokenSpeed attn_res_fwd).
// layer_residual [tokens, hidden]; block_residual [blocks, tokens, hidden]
// candidate snapshots; candidates = blocks[0..blocks) + layer_residual.
// Per token: logits_n = rms_n(layer_or_block_n) . (rms_weight * res_weight);
// out = sum_n softmax(logits)_n * v_n (FP32 accumulate), then optional fused
// output RMSNorm with out_norm_weight.
struct AttnResRequest {
  MutableTensorView layer_residual;   // in/out
  TensorView block_residual;          // [blocks, tokens, hidden]
  TensorView res_weight;              // [hidden]
  TensorView rms_weight;              // [hidden]
  std::optional<TensorView> out_norm_weight;
  float epsilon = 1.0e-5F;
  float out_norm_epsilon = 1.0e-5F;
};

// Qwen gated-residual hyperconnection.
struct HcMixRequest {
  TensorView normalized;              // [tokens, hc_count * hidden]
  TensorView projection_weight;       // [lowrank + hc_count, hc_count * hidden]
  TensorView up_weight;               // [hc_count * hidden, lowrank]
  MutableTensorView mixed;            // [tokens, hidden]
  MutableTensorView inject_logits;    // [tokens, hc_count]
  uint32_t hc_count = 4;
  uint32_t hidden_size = 2560;
  uint32_t lowrank = 320;
  float projection_scale = 1.0F;
};

struct HcCombineRequest {
  TensorView block_output;            // [tokens, hidden]
  MutableTensorView residual;         // [tokens, hc_count * hidden] in/out
  TensorView inject_logits;           // [tokens, hc_count]
  uint32_t hc_count = 4;
  uint32_t hidden_size = 2560;
};

// Matrix hyperconnection (generalized m-stream mixing).
struct MhcPreRequest {
  TensorView residual;                // [tokens, m, hidden] bf16
  TensorView fn;                      // [2m + m*m, m * hidden] fp32
  TensorView hc_scale;                // [3] fp32
  TensorView hc_base;                 // [2m + m*m] fp32
  MutableTensorView layer_input;      // [tokens, hidden] bf16
  MutableTensorView post;             // [tokens, m] fp32
  MutableTensorView comb;             // [tokens, m, m] fp32
  float rms_eps = 1.0e-5F;
  float hc_eps = 1.0e-5F;
  uint32_t sinkhorn_iters = 20;
};

struct MhcPostRequest {
  TensorView hidden_states;           // [tokens, hidden]
  MutableTensorView residual;         // [tokens, m, hidden] in/out
  TensorView post;                    // [tokens, m]
  TensorView comb;                    // [tokens, m, m]
};

// Depthwise causal short-FIR convolution with ring cache (TokenSpeed
// inkling_ring_sconv): x [tokens, width] varlen, weight [width, window],
// conv_cache [slots, ring, width] in-place ring updates.
struct RingSconvRequest {
  TensorView x;                       // [tokens, width]
  TensorView weight;                  // [width, window]
  MutableTensorView conv_cache;       // [slots, ring, width]
  TensorView seq_lens;                // [batch] int32 tokens per sequence
  TensorView cache_indices;           // [batch] int32 slot per sequence
  MutableTensorView output;           // [tokens, width]
  uint32_t window = 4;
};

[[nodiscard]] absl::Status ValidateAttnRes(const AttnResRequest& request);
[[nodiscard]] absl::Status ReferenceAttnRes(const AttnResRequest& request);
[[nodiscard]] absl::Status ValidateHcMix(const HcMixRequest& request);
[[nodiscard]] absl::Status ReferenceHcMix(const HcMixRequest& request);
[[nodiscard]] absl::Status ValidateHcCombine(const HcCombineRequest& request);
[[nodiscard]] absl::Status ReferenceHcCombine(const HcCombineRequest& request);
[[nodiscard]] absl::Status ValidateMhcPre(const MhcPreRequest& request);
[[nodiscard]] absl::Status ReferenceMhcPre(const MhcPreRequest& request);
[[nodiscard]] absl::Status ValidateMhcPost(const MhcPostRequest& request);
[[nodiscard]] absl::Status ReferenceMhcPost(const MhcPostRequest& request);
[[nodiscard]] absl::Status ValidateRingSconv(const RingSconvRequest& request);
[[nodiscard]] absl::Status ReferenceRingSconv(const RingSconvRequest& request);

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_OPS_MODEL_FUSED_CONTRACTS_H_
