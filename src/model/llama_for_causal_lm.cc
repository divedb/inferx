#include "inferx/model/llama_for_causal_lm.h"

#include <cstdint>
#include <span>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/embedding.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/logits.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/ops/rope.h"
#include "inferx/ops/swiglu.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::model {
namespace {

using ops::OpExecutor;

absl::Status Invalid(std::string_view message) {
  return absl::InvalidArgumentError(absl::StrCat("llama.forward: ", message));
}

// Same-storage flat [rows, cols] view over a contiguous mutable buffer view.
absl::StatusOr<MutableTensorView> FlatView(const MutableTensorView& source, uint64_t rows,
                                           uint64_t cols) {
  const uint64_t elements = rows * cols;
  if (elements == 0) return Invalid("flat view of an empty buffer");
  const Shape shape = Shape::Create(std::array<uint64_t, 2>{rows, cols}).value();
  const Strides strides = Strides::Contiguous(shape).value();
  return MutableTensorView::Create(source.buffer(), source.dtype(), shape, strides,
                                   source.byte_offset());
}

void Trace(IntermediateTraceSink* sink, const std::string& name, const TensorView& value) {
  if (sink != nullptr) sink->OnIntermediate(name, value);
}

ops::GemmRequest Linear(const TensorView& input, const TensorView& weight,
                        const MutableTensorView& output) {
  ops::GemmRequest request;
  request.input = input;
  request.weight = weight;
  request.output = output;
  return request;
}

}  // namespace

absl::StatusOr<LlamaForCausalLM> LlamaForCausalLM::Create(ModelSpec spec) {
  const LlamaSpec& llama = spec.llama();
  if (llama.vocab_size == 0 || llama.hidden_size == 0 || llama.intermediate_size == 0 ||
      llama.num_hidden_layers == 0 || llama.num_attention_heads == 0 ||
      llama.num_key_value_heads == 0 || llama.head_dim == 0 || llama.max_position_embeddings == 0) {
    return absl::FailedPreconditionError("llama.spec: incomplete model specification");
  }
  if (llama.num_attention_heads % llama.num_key_value_heads != 0) {
    return absl::FailedPreconditionError("llama.spec: query heads must divide key-value heads");
  }
  return LlamaForCausalLM(std::move(spec));
}

absl::Status LlamaForCausalLM::Forward(const ForwardBatch& batch, const ModelWeights& weights,
                                       OpExecutor& executor) const {
  const LlamaSpec& llama = spec_.llama();
  const uint64_t tokens = batch.num_tokens;
  const uint64_t heads = llama.num_attention_heads;
  const uint64_t kv_heads = llama.num_key_value_heads;
  const uint64_t head_dim = llama.head_dim;

  if (batch.activations == nullptr) return Invalid("missing activation buffers");
  if (batch.work == WorkKind::kDecode && tokens != 1) {
    return Invalid("decode steps carry exactly one token");
  }
  if (batch.token_ids.shape().dim(0) != tokens || batch.positions.shape().dim(0) != tokens ||
      batch.host_token_ids.size() != tokens || batch.host_positions.size() != tokens) {
    return Invalid("token/position counts disagree");
  }
  if (batch.key_cache.size() != llama.num_hidden_layers ||
      batch.value_cache.size() != llama.num_hidden_layers) {
    return Invalid("per-layer KV cache views missing");
  }
  if (weights.layers().size() != llama.num_hidden_layers) {
    return Invalid("weight table layer count mismatch");
  }
  for (const int32_t token : batch.host_token_ids) {
    if (token < 0 || static_cast<uint64_t>(token) >= llama.vocab_size) {
      return Invalid("token id outside the model vocabulary");
    }
  }
  for (const int32_t position : batch.host_positions) {
    if (position < 0 || static_cast<uint64_t>(position) >= llama.max_position_embeddings) {
      return Invalid("position outside the model context");
    }
  }
  if (kv_heads * head_dim == 0 || heads * head_dim == 0) return Invalid("zero head geometry");

  const ActivationBuffers& act = *batch.activations;

  ops::EmbeddingRequest embedding{batch.token_ids, weights.embedding(), act.hidden,
                                  batch.host_token_ids};
  absl::Status status = executor.Embedding(embedding);
  if (!status.ok()) return status;
  Trace(batch.trace, "embedding", act.hidden.AsConst());

  // Batch-one attention metadata: one sequence, `tokens` new entries, the
  // cache holds kv_append_before prior committed positions.
  const int32_t indptr[2] = {0, static_cast<int32_t>(tokens)};
  const int32_t lengths_before[1] = {static_cast<int32_t>(batch.kv_append_begin)};
  const ops::ExecutionPhase phase = batch.work == WorkKind::kPrefill ? ops::ExecutionPhase::kPrefill
                                                                     : ops::ExecutionPhase::kDecode;

  for (uint32_t layer = 0; layer < llama.num_hidden_layers; ++layer) {
    const LayerWeights& weight = weights.layers()[layer];
    const std::string prefix = absl::StrCat("layers.", layer, ".");

    status = executor.RmsNorm({act.hidden.AsConst(), weight.input_norm, act.normed,
                               static_cast<float>(llama.rms_norm_eps)});
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "input_norm", act.normed.AsConst());

    absl::StatusOr<MutableTensorView> query_flat = FlatView(act.query, tokens, heads * head_dim);
    absl::StatusOr<MutableTensorView> key_flat = FlatView(act.key, tokens, kv_heads * head_dim);
    absl::StatusOr<MutableTensorView> value_flat = FlatView(act.value, tokens, kv_heads * head_dim);
    absl::StatusOr<MutableTensorView> attn_flat = FlatView(act.attn_out, tokens, heads * head_dim);
    if (!query_flat.ok() || !key_flat.ok() || !value_flat.ok() || !attn_flat.ok()) {
      return Invalid("activation view geometry");
    }

    status = executor.Gemm(Linear(act.normed.AsConst(), weight.query, *query_flat));
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "q_proj", query_flat->AsConst());
    status = executor.Gemm(Linear(act.normed.AsConst(), weight.key, *key_flat));
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "k_proj", key_flat->AsConst());
    status = executor.Gemm(Linear(act.normed.AsConst(), weight.value, *value_flat));
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "v_proj", value_flat->AsConst());

    // In-place RoPE (exact alias): cached K is rotated exactly once.
    status = executor.Rope({act.query.AsConst(), act.key.AsConst(), batch.positions, act.query,
                            act.key, static_cast<float>(llama.rope_theta),
                            llama.max_position_embeddings, batch.host_positions});
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "q_rope", act.query.AsConst());
    Trace(batch.trace, prefix + "k_rope", act.key.AsConst());

    ops::AttentionRequest attention{act.query.AsConst(),
                                    act.key.AsConst(),
                                    act.value.AsConst(),
                                    std::span<const int32_t>(indptr, 2),
                                    std::span<const int32_t>(indptr, 2),
                                    batch.host_positions,
                                    std::span<const int32_t>(lengths_before, 1),
                                    batch.key_cache[layer],
                                    batch.value_cache[layer],
                                    act.attn_out,
                                    phase};
    status = executor.Attention(attention);
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "attention", act.attn_out.AsConst());

    status = executor.Gemm(Linear(attn_flat->AsConst(), weight.attention_output, act.projected));
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "o_proj", act.projected.AsConst());

    // Exact-left in-place residual: `hidden` keeps the pre-attention value of
    // the left operand while being overwritten with the sum.
    status = executor.Residual({act.hidden.AsConst(), act.projected.AsConst(), act.hidden});
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "post_attention_residual", act.hidden.AsConst());

    status = executor.RmsNorm({act.hidden.AsConst(), weight.post_norm, act.normed,
                               static_cast<float>(llama.rms_norm_eps)});
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "post_attention_norm", act.normed.AsConst());

    status = executor.Gemm(Linear(act.normed.AsConst(), weight.mlp_gate, act.gate));
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "mlp_gate", act.gate.AsConst());
    status = executor.Gemm(Linear(act.normed.AsConst(), weight.mlp_up, act.up));
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "mlp_up", act.up.AsConst());

    status = executor.SwiGlu({act.gate.AsConst(), act.up.AsConst(), act.activated});
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "mlp_activated", act.activated.AsConst());

    status = executor.Gemm(Linear(act.activated.AsConst(), weight.mlp_down, act.down_out));
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "mlp_down", act.down_out.AsConst());

    status = executor.Residual({act.hidden.AsConst(), act.down_out.AsConst(), act.hidden});
    if (!status.ok()) return status;
    Trace(batch.trace, prefix + "output", act.hidden.AsConst());
  }

  status = executor.RmsNorm({act.hidden.AsConst(), weights.final_norm(), act.final_norm,
                             static_cast<float>(llama.rms_norm_eps)});
  if (!status.ok()) return status;
  Trace(batch.trace, "final_norm", act.final_norm.AsConst());

  // Ordinary generation computes the final logical row only.
  absl::StatusOr<TensorView> selected = act.final_norm.AsConst().Slice(0, tokens - 1, tokens);
  if (!selected.ok()) return selected.status();
  status = executor.Logits({*selected, weights.lm_head(), act.logits});
  if (!status.ok()) return status;
  Trace(batch.trace, "logits", act.logits.AsConst());

  return absl::OkStatus();
}

}  // namespace inferx::model
