#include "inferx/runtime/model_memory_plan.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/artifacts/artifact_tensor.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/token.h"
#include "inferx/model/model_spec.h"
#include "inferx/model/parameter_spec.h"
#include "inferx/model/weight_plan.h"
#include "inferx/tensor/dtype.h"

namespace inferx::runtime {
namespace {

absl::Status PlanError(std::string_view field, std::string_view detail) {
  return absl::InvalidArgumentError(absl::StrCat("memory_plan.", field, ": ", detail));
}

absl::StatusOr<uint64_t> ArtifactDtypeSize(const artifacts::ArtifactDtype dtype) {
  switch (dtype) {
    case artifacts::ArtifactDtype::kF16:
    case artifacts::ArtifactDtype::kBf16:
      return uint64_t{2};
    case artifacts::ArtifactDtype::kF32:
      return uint64_t{4};
    default:
      return PlanError("dtype", "unsupported weight dtype for M5 execution");
  }
}

}  // namespace

absl::StatusOr<ModelMemoryPlan> ModelMemoryPlanner::Plan(const MemoryPlanRequest& request) {
  if (request.spec == nullptr || request.weight_plan == nullptr) {
    return PlanError("request", "spec, parameters, and weight plan are required");
  }
  const model::LlamaSpec& llama = request.spec->llama();
  if (request.max_prefill_tokens == 0 || request.context_capacity == 0) {
    return PlanError("envelope", "prefill/context limits must be positive");
  }
  if (request.context_capacity < request.max_prefill_tokens ||
      request.context_capacity > llama.max_position_embeddings) {
    return PlanError("envelope", "context capacity must cover prefill and fit the model maximum");
  }

  ModelMemoryPlan plan;
  plan.max_prefill_tokens = request.max_prefill_tokens;
  plan.context_capacity = request.context_capacity;

  // Weight arena: canonical ParameterId order over the weight plan; tied
  // aliases place no bytes. Bytes/alignment follow the execution dtype: the
  // M5 CPU family materializes every floating source (F16/BF16/F32) as FP32
  // (m5.md section 16.1), and the source dtype is only validated, not kept.
  constexpr uint64_t kExecutionBytes = 4;  // FP32 family
  uint64_t offset = 0;
  for (const model::WeightPlanItem& item : request.weight_plan->items) {
    const absl::StatusOr<uint64_t> source_size = ArtifactDtypeSize(item.source.dtype);
    if (!source_size.ok()) return source_size.status();
    uint64_t elements = 1;
    for (const uint64_t dim : item.source.shape) {
      const auto multiplied = CheckedMul(elements, dim);
      if (!multiplied.ok()) return PlanError("weight", "element count overflow");
      elements = *multiplied;
    }
    const auto bytes = CheckedMul(elements, kExecutionBytes);
    if (!bytes.ok()) return PlanError("weight", "byte count overflow");
    const uint64_t alignment = std::max(request.weight_alignment, kExecutionBytes);
    const auto aligned = CheckedAdd(offset, alignment - 1);
    if (!aligned.ok()) return PlanError("weight", "offset overflow");
    const uint64_t placement = (*aligned / alignment) * alignment;
    const auto end = CheckedAdd(placement, *bytes);
    if (!end.ok()) return PlanError("weight", "arena overflow");
    plan.weights.push_back(
        {item.parameter, placement, *bytes, alignment, placement - offset, false});
    plan.weight_padding_bytes += placement - offset;
    offset = *end;
  }
  for (const model::ParameterAlias& alias : request.weight_plan->aliases) {
    plan.weights.push_back({alias.parameter, 0, 0, 0, 0, true});
  }
  plan.weight_arena_bytes = offset;

  // Contiguous KV: one backing range, separately aligned K and V arenas,
  // per-layer views [1, C, Nkv, D] (m5.md section 9.3).
  const uint64_t elements_per_kind = static_cast<uint64_t>(llama.num_hidden_layers) *
                                     request.context_capacity * llama.num_key_value_heads *
                                     llama.head_dim;
  const auto doubled = CheckedMul(elements_per_kind, uint64_t{2});
  if (!doubled.ok()) return PlanError("kv", "element count overflow");
  const auto kv_bytes = CheckedMul(*doubled, kExecutionBytes);
  if (!kv_bytes.ok()) return PlanError("kv", "byte count overflow");
  plan.kv_bytes = *kv_bytes;
  plan.kv_alignment = 256;

  // Activations for the maximum prefill shape, FP32 family, plus the final
  // row logits. Element counts only; the backend assigns arena offsets.
  const uint64_t t = request.max_prefill_tokens;
  const uint64_t h = llama.hidden_size;
  const uint64_t qkv = llama.num_key_value_heads * llama.head_dim;
  const uint64_t q = llama.num_attention_heads * llama.head_dim;
  const uint64_t i = llama.intermediate_size;
  const std::vector<std::pair<std::string_view, uint64_t>> buffers = {
      {"hidden", t * h},  {"normed", t * h},    {"query", t * q},     {"key", t * qkv},
      {"value", t * qkv}, {"attn_out", t * q},  {"projected", t * h}, {"gate", t * i},
      {"up", t * i},      {"activated", t * i}, {"down_out", t * h},  {"final_norm", t * h},
  };
  uint64_t activation_elements = 0;
  for (const auto& [name, elements] : buffers) {
    const auto total = CheckedAdd(activation_elements, elements);
    if (!total.ok()) return PlanError("activation", "element count overflow");
    activation_elements = *total;
    plan.activations.push_back({std::string(name), elements});
  }
  const auto activation_bytes = CheckedMul(activation_elements, uint64_t{4});
  if (!activation_bytes.ok()) return PlanError("activation", "byte count overflow");
  plan.activation_bytes = *activation_bytes;

  const auto logits_bytes = CheckedMul(llama.vocab_size, uint64_t{4});
  if (!logits_bytes.ok()) return PlanError("logits", "byte count overflow");
  plan.logits_bytes = *logits_bytes;
  plan.host_logits_bytes = *logits_bytes;
  plan.execution_dtype = Dtype::kFloat32;

  const auto total = CheckedAdd(plan.weight_arena_bytes, plan.kv_bytes);
  if (!total.ok()) return PlanError("total", "device byte overflow");
  const auto total2 = CheckedAdd(*total, plan.activation_bytes);
  if (!total2.ok()) return PlanError("total", "device byte overflow");
  const auto total3 = CheckedAdd(*total2, plan.logits_bytes);
  if (!total3.ok()) return PlanError("total", "device byte overflow");
  plan.total_device_bytes = *total3;

  return plan;
}

}  // namespace inferx::runtime
