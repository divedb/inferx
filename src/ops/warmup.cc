#include "inferx/ops/warmup.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/checked_math.h"

namespace inferx::ops {
namespace {

KernelKey BaseKey(OpKind op, const OperatorEnvelope& envelope, uint8_t rank) {
  KernelKey key;
  key.op = op;
  key.device_kind = envelope.device_kind;
  key.compute_capability = envelope.compute_capability;
  key.input_dtype = envelope.storage_dtype;
  key.weight_dtype = envelope.storage_dtype;
  key.output_dtype = envelope.storage_dtype;
  key.rank = rank;
  key.workspace_limit_bytes = envelope.workspace_limit_bytes;
  key.alignment_class = envelope.device_kind == DeviceKind::kCuda ? 16 : alignof(float);
  return key;
}

void AddGemm(std::vector<KernelKey>* keys, const OperatorEnvelope& envelope, uint32_t tokens,
             uint64_t inner, uint64_t output) {
  KernelKey key = BaseKey(OpKind::kGemm, envelope, 3);
  key.dimensions[0] = tokens;
  key.dimensions[1] = inner;
  key.dimensions[2] = output;
  keys->push_back(key);
}

}  // namespace

absl::Status RegisterReferenceCapabilities(KernelRegistry& registry) {
  auto add = [&](OpKind op, uint8_t rank, Dtype input_dtype, LayoutId input_layout,
                 LayoutId weight_layout, LayoutId output_layout, uint8_t phase_mask,
                 uint8_t alias_mask) -> absl::Status {
    BackendCapability capability;
    capability.backend = BackendId::kReference;
    capability.capability_id = std::string("reference.") + std::string(OpKindName(op)) + ".fp32";
    capability.priority = 10'000;
    capability.op = op;
    capability.device_kind = DeviceKind::kHost;
    capability.phase_mask = phase_mask;
    capability.input_dtype_mask = DtypeMask(input_dtype);
    capability.weight_dtype_mask = DtypeMask(Dtype::kFloat32);
    capability.output_dtype_mask = DtypeMask(Dtype::kFloat32);
    capability.input_layout = input_layout;
    capability.weight_layout = weight_layout;
    capability.output_layout = output_layout;
    capability.alias_mask = alias_mask;
    capability.rank = rank;
    for (size_t axis = 0; axis < rank; ++axis) {
      capability.dimensions[axis] = DimensionRange{0, std::numeric_limits<uint64_t>::max()};
    }
    capability.maximum_query_heads = 64;
    capability.maximum_kv_heads = 64;
    capability.maximum_head_dimension = 256;
    capability.maximum_sequence_bucket = 4096;
    capability.minimum_operand_alignment = alignof(float);
    return registry.Register(std::move(capability));
  };

  constexpr uint8_t kAllPhases = PhaseMask(ExecutionPhase::kGeneric) |
                                 PhaseMask(ExecutionPhase::kPrefill) |
                                 PhaseMask(ExecutionPhase::kDecode);
  constexpr uint8_t kAttentionPhases =
      PhaseMask(ExecutionPhase::kPrefill) | PhaseMask(ExecutionPhase::kDecode);
  constexpr uint8_t kDisjoint = AliasMask(AliasMode::kDisjoint);
  constexpr uint8_t kUnaryAliases = kDisjoint | AliasMask(AliasMode::kExactInPlace);
  constexpr uint8_t kBinaryAliases =
      kDisjoint | AliasMask(AliasMode::kExactLeft) | AliasMask(AliasMode::kExactRight);

  absl::Status status =
      add(OpKind::kEmbedding, 2, Dtype::kInt32, LayoutId::kRowMajorDense, LayoutId::kRowMajorDense,
          LayoutId::kRowMajorDense, kAllPhases, kDisjoint);
  if (!status.ok()) return status;
  for (OpKind op : {OpKind::kGemm, OpKind::kRmsNorm, OpKind::kLogits}) {
    const uint8_t rank = op == OpKind::kRmsNorm ? 2 : 3;
    status = add(op, rank, Dtype::kFloat32, LayoutId::kRowMajorDense, LayoutId::kRowMajorDense,
                 LayoutId::kRowMajorDense, kAllPhases, kDisjoint);
    if (!status.ok()) return status;
  }
  status =
      add(OpKind::kRope, 4, Dtype::kFloat32, LayoutId::kQkvTokenHeadDim, LayoutId::kRowMajorDense,
          LayoutId::kQkvTokenHeadDim, kAllPhases, kDisjoint | kUnaryAliases);
  if (!status.ok()) return status;
  for (OpKind op : {OpKind::kSilu, OpKind::kMultiply, OpKind::kSiluMultiply, OpKind::kResidual}) {
    const uint8_t aliases = op == OpKind::kSilu ? kUnaryAliases : kBinaryAliases;
    status = add(op, 2, Dtype::kFloat32, LayoutId::kRowMajorDense, LayoutId::kRowMajorDense,
                 LayoutId::kRowMajorDense, kAllPhases, aliases);
    if (!status.ok()) return status;
  }
  status =
      add(OpKind::kAttention, 4, Dtype::kFloat32, LayoutId::kQkvTokenHeadDim,
          LayoutId::kContiguousKvBshd, LayoutId::kQkvTokenHeadDim, kAttentionPhases, kDisjoint);
  if (!status.ok()) return status;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<KernelKey>> BuildRequiredKernelSet(const LlamaOperatorSpec& llama,
                                                              const OperatorEnvelope& envelope) {
  if (envelope.device_kind > DeviceKind::kCuda ||
      (envelope.device_kind == DeviceKind::kHost && envelope.compute_capability != 0) ||
      (envelope.device_kind == DeviceKind::kCuda && envelope.compute_capability == 0)) {
    return absl::InvalidArgumentError("operator_envelope.device: invalid device/SM combination");
  }
  if (envelope.maximum_batch == 0 || envelope.maximum_context == 0 ||
      envelope.token_buckets.empty() || llama.hidden_size == 0 || llama.intermediate_size == 0 ||
      llama.vocab_size == 0 || llama.num_attention_heads == 0 || llama.num_key_value_heads == 0 ||
      llama.head_dim == 0 || llama.num_attention_heads % llama.num_key_value_heads != 0) {
    return absl::InvalidArgumentError("operator_envelope: invalid model or finite envelope");
  }
  if (envelope.device_kind == DeviceKind::kHost && envelope.storage_dtype != Dtype::kFloat32) {
    return absl::UnimplementedError("operator_envelope: CPU reference accepts FP32 storage only");
  }
  if (envelope.device_kind == DeviceKind::kCuda && envelope.storage_dtype != Dtype::kFloat32 &&
      envelope.storage_dtype != Dtype::kFloat16 && envelope.storage_dtype != Dtype::kBFloat16) {
    return absl::UnimplementedError("operator_envelope: CUDA accepts FP32, FP16, or BF16 storage");
  }
  std::vector<KernelKey> keys;
  keys.reserve(envelope.token_buckets.size() * 12U + 2U);
  for (uint32_t tokens : envelope.token_buckets) {
    if (tokens > envelope.maximum_prompt_tokens) {
      return absl::InvalidArgumentError("operator_envelope.token_buckets: bucket exceeds maximum");
    }
    KernelKey embedding = BaseKey(OpKind::kEmbedding, envelope, 2);
    embedding.input_dtype = Dtype::kInt32;
    embedding.dimensions[0] = tokens;
    embedding.dimensions[1] = llama.hidden_size;
    keys.push_back(embedding);

    KernelKey norm = BaseKey(OpKind::kRmsNorm, envelope, 2);
    norm.dimensions[0] = tokens;
    norm.dimensions[1] = llama.hidden_size;
    keys.push_back(norm);

    const uint64_t query_width = static_cast<uint64_t>(llama.num_attention_heads) * llama.head_dim;
    const uint64_t kv_width = static_cast<uint64_t>(llama.num_key_value_heads) * llama.head_dim;
    AddGemm(&keys, envelope, tokens, llama.hidden_size, query_width);
    AddGemm(&keys, envelope, tokens, llama.hidden_size, kv_width);
    AddGemm(&keys, envelope, tokens, query_width, llama.hidden_size);
    AddGemm(&keys, envelope, tokens, llama.hidden_size, llama.intermediate_size);
    AddGemm(&keys, envelope, tokens, llama.intermediate_size, llama.hidden_size);

    KernelKey rope = BaseKey(OpKind::kRope, envelope, 4);
    rope.dimensions[0] = tokens;
    rope.dimensions[1] = llama.num_attention_heads;
    rope.dimensions[2] = llama.num_key_value_heads;
    rope.dimensions[3] = llama.head_dim;
    rope.input_layout = LayoutId::kQkvTokenHeadDim;
    rope.output_layout = LayoutId::kQkvTokenHeadDim;
    keys.push_back(rope);

    KernelKey activation = BaseKey(OpKind::kSiluMultiply, envelope, 2);
    activation.dimensions[0] = tokens;
    activation.dimensions[1] = llama.intermediate_size;
    keys.push_back(activation);

    KernelKey residual = BaseKey(OpKind::kResidual, envelope, 2);
    residual.dimensions[0] = tokens;
    residual.dimensions[1] = llama.hidden_size;
    residual.alias_mode = AliasMode::kExactLeft;
    keys.push_back(residual);

    KernelKey attention = BaseKey(OpKind::kAttention, envelope, 4);
    attention.phase = tokens == 1 ? ExecutionPhase::kDecode : ExecutionPhase::kPrefill;
    attention.input_layout = LayoutId::kQkvTokenHeadDim;
    attention.weight_layout = LayoutId::kContiguousKvBshd;
    attention.output_layout = LayoutId::kQkvTokenHeadDim;
    attention.dimensions[0] = envelope.maximum_batch;
    attention.dimensions[1] = envelope.maximum_context;
    attention.dimensions[2] = llama.num_attention_heads;
    attention.dimensions[3] = llama.head_dim;
    attention.query_heads = llama.num_attention_heads;
    attention.kv_heads = llama.num_key_value_heads;
    attention.head_dimension = llama.head_dim;
    attention.sequence_bucket = tokens;
    attention.causal = true;
    keys.push_back(attention);

    KernelKey logits = BaseKey(OpKind::kLogits, envelope, 3);
    logits.output_dtype = Dtype::kFloat32;
    logits.dimensions[0] = tokens;
    logits.dimensions[1] = llama.hidden_size;
    logits.dimensions[2] = llama.vocab_size;
    keys.push_back(logits);
  }
  std::sort(keys.begin(), keys.end(), KernelKeyLess);
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

absl::StatusOr<PreparedKernelCache> WarmupKernelSet(KernelRegistry& registry,
                                                    std::span<const KernelKey> required_keys,
                                                    size_t cache_capacity,
                                                    const PrepareKernel& prepare,
                                                    std::optional<BackendId> forced_backend) {
  if (!prepare) return absl::InvalidArgumentError("warmup.prepare: callback is required");
  if (cache_capacity < required_keys.size()) {
    return absl::ResourceExhaustedError("warmup.cache: capacity is smaller than required key set");
  }
  if (!registry.frozen()) {
    absl::Status status = registry.Freeze();
    if (!status.ok()) return status;
  }
  PreparedKernelCache transaction(cache_capacity);
  for (const KernelKey& key : required_keys) {
    absl::StatusOr<KernelSelection> selection = registry.Select(key, forced_backend);
    if (!selection.ok()) return selection.status();
    absl::StatusOr<PreparedKernelPlan> plan = prepare(*selection, key);
    if (!plan.ok()) return plan.status();
    if (plan->key != key || plan->backend != selection->backend ||
        plan->capability_id != selection->capability_id ||
        plan->workspace_bytes > key.workspace_limit_bytes) {
      return absl::InternalError("warmup.prepare: prepared plan does not match selection");
    }
    absl::Status status = transaction.Insert(std::move(*plan));
    if (!status.ok()) return status;
  }
  absl::Status status = transaction.Freeze();
  if (!status.ok()) return status;
  return transaction;
}

}  // namespace inferx::ops
