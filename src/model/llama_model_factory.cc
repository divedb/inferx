#include "inferx/model/llama_model_factory.h"

#include <simdjson.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../artifacts/json_internal.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace inferx::model {
namespace {

using artifacts::internal::JsonError;

struct OptionalElement {
  bool present = false;
  simdjson::dom::element value;
};

absl::StatusOr<OptionalElement> Optional(simdjson::dom::object object, std::string_view key) {
  simdjson::dom::element value;
  const auto error = object.at_key(key).get(value);
  if (error == simdjson::NO_SUCH_FIELD) return OptionalElement{};
  if (error) return JsonError("", simdjson::error_message(error));
  return OptionalElement{true, value};
}

absl::StatusOr<uint64_t> RequiredPositive(simdjson::dom::object object, std::string_view key) {
  auto field = artifacts::internal::Required(object, key, "");
  if (!field.ok()) return field.status();
  auto value = artifacts::internal::Uint64(*field, absl::StrCat("/", key));
  if (!value.ok()) return value.status();
  if (*value == 0) return JsonError(absl::StrCat("/", key), "must be positive");
  return value;
}

absl::StatusOr<uint64_t> OptionalPositive(simdjson::dom::object object, std::string_view key,
                                          uint64_t default_value) {
  auto field = Optional(object, key);
  if (!field.ok()) return field.status();
  if (!field->present) return default_value;
  auto value = artifacts::internal::Uint64(field->value, absl::StrCat("/", key));
  if (!value.ok()) return value.status();
  if (*value == 0) return JsonError(absl::StrCat("/", key), "must be positive");
  return value;
}

absl::StatusOr<bool> OptionalBool(simdjson::dom::object object, std::string_view key,
                                  bool default_value) {
  auto field = Optional(object, key);
  if (!field.ok()) return field.status();
  if (!field->present) return default_value;
  return artifacts::internal::Bool(field->value, absl::StrCat("/", key));
}

absl::StatusOr<std::optional<int32_t>> OptionalToken(simdjson::dom::object object,
                                                     std::string_view key, uint64_t vocab_size) {
  auto field = Optional(object, key);
  if (!field.ok()) return field.status();
  if (!field->present || field->value.is_null()) return std::nullopt;
  auto value = artifacts::internal::Uint64(field->value, absl::StrCat("/", key));
  if (!value.ok()) return value.status();
  if (*value >= vocab_size || *value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return JsonError(absl::StrCat("/", key), "token ID is outside the model vocabulary");
  }
  return static_cast<int32_t>(*value);
}

absl::Status ValidateAbsentOrNull(simdjson::dom::object object, std::string_view key) {
  auto field = Optional(object, key);
  if (!field.ok()) return field.status();
  if (field->present && !field->value.is_null()) {
    return absl::UnimplementedError(
        absl::StrCat("/", key, ": recognized feature is unsupported in M3"));
  }
  return absl::OkStatus();
}

ParameterSpec MakeParameter(std::string name, ParameterRole role, artifacts::ArtifactShape shape,
                            std::optional<uint32_t> layer = std::nullopt) {
  return ParameterSpec{ParameterId(0),
                       std::move(name),
                       role,
                       std::move(shape),
                       {artifacts::ArtifactDType::kF16, artifacts::ArtifactDType::kBf16,
                        artifacts::ArtifactDType::kF32},
                       layer,
                       std::nullopt};
}

}  // namespace

absl::StatusOr<ModelSpec> LlamaModelFactory::ParseConfig(
    const artifacts::ArtifactFile& config, const artifacts::ArtifactLimits& limits) const {
  auto bytes = config.ReadAll(limits.max_json_bytes);
  if (!bytes.ok()) return bytes.status();
  simdjson::dom::parser parser;
  if (const auto error = parser.allocate(bytes->size(), limits.max_json_depth); error) {
    return artifacts::internal::ParseError(config.relative_path(), error);
  }
  simdjson::dom::element document;
  const auto parse_error =
      parser.parse(reinterpret_cast<const uint8_t*>(bytes->data()), bytes->size(), true)
          .get(document);
  if (parse_error) {
    return artifacts::internal::ParseError(config.relative_path(), parse_error);
  }
  auto root = artifacts::internal::Object(document, "");
  if (!root.ok()) return root.status();
  if (auto status = artifacts::internal::CheckUniqueKeys(*root, ""); !status.ok()) {
    return status;
  }

  const std::set<std::string> consumed = {"architectures",
                                          "attention_bias",
                                          "attention_dropout",
                                          "bos_token_id",
                                          "eos_token_id",
                                          "head_dim",
                                          "hidden_act",
                                          "hidden_size",
                                          "intermediate_size",
                                          "max_position_embeddings",
                                          "mlp_bias",
                                          "model_type",
                                          "num_attention_heads",
                                          "num_hidden_layers",
                                          "num_key_value_heads",
                                          "pad_token_id",
                                          "pretraining_tp",
                                          "quantization_config",
                                          "rms_norm_eps",
                                          "rope_scaling",
                                          "rope_theta",
                                          "sliding_window",
                                          "tie_word_embeddings",
                                          "torch_dtype",
                                          "vocab_size"};
  const std::set<std::string> inert = {"_name_or_path",
                                       "bos_token",
                                       "eos_token",
                                       "initializer_range",
                                       "output_attentions",
                                       "output_hidden_states",
                                       "pad_token",
                                       "problem_type",
                                       "pruned_heads",
                                       "return_dict",
                                       "task_specific_params",
                                       "torchscript",
                                       "transformers_version",
                                       "unk_token",
                                       "use_cache"};
  for (const auto field : *root) {
    if (!consumed.contains(std::string(field.key)) && !inert.contains(std::string(field.key))) {
      return absl::UnimplementedError(
          absl::StrCat("/", field.key, ": unclassified Llama config field"));
    }
  }

  auto model_type_value = artifacts::internal::Required(*root, "model_type", "");
  if (!model_type_value.ok()) return model_type_value.status();
  auto model_type = artifacts::internal::String(*model_type_value, "/model_type");
  if (!model_type.ok()) return model_type.status();
  if (*model_type != "llama") {
    return absl::UnimplementedError("only model_type=llama is supported");
  }
  auto architectures = Optional(*root, "architectures");
  if (!architectures.ok()) return architectures.status();
  if (architectures->present) {
    auto values = artifacts::internal::Array(architectures->value, "/architectures");
    if (!values.ok()) return values.status();
    if (values->size() != 1) {
      return absl::UnimplementedError("/architectures must select exactly LlamaForCausalLM");
    }
    for (simdjson::dom::element value : *values) {
      auto architecture = artifacts::internal::String(value, "/architectures/0");
      if (!architecture.ok()) return architecture.status();
      if (*architecture != "LlamaForCausalLM") {
        return absl::UnimplementedError("unsupported Llama architecture");
      }
    }
  }

  auto vocab_size = RequiredPositive(*root, "vocab_size");
  if (!vocab_size.ok()) return vocab_size.status();
  auto hidden_size = RequiredPositive(*root, "hidden_size");
  if (!hidden_size.ok()) return hidden_size.status();
  auto intermediate_size = RequiredPositive(*root, "intermediate_size");
  if (!intermediate_size.ok()) return intermediate_size.status();
  auto layer_count = RequiredPositive(*root, "num_hidden_layers");
  if (!layer_count.ok()) return layer_count.status();
  auto attention_heads = RequiredPositive(*root, "num_attention_heads");
  if (!attention_heads.ok()) return attention_heads.status();
  auto kv_heads = OptionalPositive(*root, "num_key_value_heads", *attention_heads);
  if (!kv_heads.ok()) return kv_heads.status();
  constexpr uint64_t kMaxModelDimension = 16ULL * 1024 * 1024;
  constexpr uint64_t kMaxModelLayers = 4096;
  constexpr uint64_t kMaxAttentionHeads = 1ULL * 1024 * 1024;
  constexpr uint64_t kMaxModelContext = 16ULL * 1024 * 1024;
  if (*hidden_size > kMaxModelDimension || *intermediate_size > kMaxModelDimension ||
      *layer_count > kMaxModelLayers || *attention_heads > kMaxAttentionHeads ||
      *kv_heads > kMaxAttentionHeads) {
    return absl::ResourceExhaustedError("Llama dimensions exceed the M3 model resource limits");
  }
  if (*vocab_size > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      *layer_count > std::numeric_limits<uint32_t>::max() ||
      *attention_heads > std::numeric_limits<uint32_t>::max() ||
      *kv_heads > std::numeric_limits<uint32_t>::max()) {
    return absl::OutOfRangeError("Llama config count exceeds runtime type");
  }
  if (*hidden_size % *attention_heads != 0) {
    return absl::InvalidArgumentError("hidden_size must be divisible by num_attention_heads");
  }
  auto head_dim = OptionalPositive(*root, "head_dim", *hidden_size / *attention_heads);
  if (!head_dim.ok()) return head_dim.status();
  if (*head_dim > std::numeric_limits<uint32_t>::max() ||
      *attention_heads > std::numeric_limits<uint64_t>::max() / *head_dim ||
      *attention_heads * *head_dim != *hidden_size) {
    return absl::InvalidArgumentError("num_attention_heads * head_dim must equal hidden_size");
  }
  if (*attention_heads % *kv_heads != 0) {
    return absl::InvalidArgumentError("num_key_value_heads must divide num_attention_heads");
  }
  auto max_positions = RequiredPositive(*root, "max_position_embeddings");
  if (!max_positions.ok()) return max_positions.status();
  if (*max_positions > kMaxModelContext) {
    return absl::ResourceExhaustedError(
        "max_position_embeddings exceeds the M3 model context limit");
  }

  auto eps_value = artifacts::internal::Required(*root, "rms_norm_eps", "");
  if (!eps_value.ok()) return eps_value.status();
  auto eps = artifacts::internal::FiniteDouble(*eps_value, "/rms_norm_eps");
  if (!eps.ok()) return eps.status();
  if (*eps <= 0) return JsonError("/rms_norm_eps", "must be positive");
  double rope_theta = 10'000.0;
  auto rope_value = Optional(*root, "rope_theta");
  if (!rope_value.ok()) return rope_value.status();
  if (rope_value->present) {
    auto value = artifacts::internal::FiniteDouble(rope_value->value, "/rope_theta");
    if (!value.ok()) return value.status();
    if (*value <= 0) return JsonError("/rope_theta", "must be positive");
    rope_theta = *value;
  }
  double dropout = 0;
  auto dropout_value = Optional(*root, "attention_dropout");
  if (!dropout_value.ok()) return dropout_value.status();
  if (dropout_value->present) {
    auto value = artifacts::internal::FiniteDouble(dropout_value->value, "/attention_dropout");
    if (!value.ok()) return value.status();
    if (*value < 0 || *value > 1) {
      return JsonError("/attention_dropout", "must be in [0, 1]");
    }
    dropout = *value;
  }

  auto hidden_act = Optional(*root, "hidden_act");
  if (!hidden_act.ok()) return hidden_act.status();
  if (hidden_act->present) {
    auto value = artifacts::internal::String(hidden_act->value, "/hidden_act");
    if (!value.ok()) return value.status();
    if (*value != "silu") {
      return absl::UnimplementedError("only hidden_act=silu is supported");
    }
  }
  auto tie = OptionalBool(*root, "tie_word_embeddings", false);
  if (!tie.ok()) return tie.status();
  for (const std::string_view key : {"attention_bias", "mlp_bias"}) {
    auto value = OptionalBool(*root, key, false);
    if (!value.ok()) return value.status();
    if (*value) {
      return absl::UnimplementedError(
          absl::StrCat("/", key, ": biased Llama weights are unsupported"));
    }
  }
  for (const std::string_view key : {"rope_scaling", "sliding_window", "quantization_config"}) {
    if (auto status = ValidateAbsentOrNull(*root, key); !status.ok()) {
      return status;
    }
  }
  auto pretraining_tp = Optional(*root, "pretraining_tp");
  if (!pretraining_tp.ok()) return pretraining_tp.status();
  if (pretraining_tp->present) {
    auto value = artifacts::internal::Uint64(pretraining_tp->value, "/pretraining_tp");
    if (!value.ok()) return value.status();
    if (*value != 1) {
      return absl::UnimplementedError("pretraining_tp must be absent or 1");
    }
  }

  auto bos = OptionalToken(*root, "bos_token_id", *vocab_size);
  if (!bos.ok()) return bos.status();
  auto pad = OptionalToken(*root, "pad_token_id", *vocab_size);
  if (!pad.ok()) return pad.status();
  std::vector<int32_t> eos;
  auto eos_value = Optional(*root, "eos_token_id");
  if (!eos_value.ok()) return eos_value.status();
  if (eos_value->present && !eos_value->value.is_null()) {
    uint64_t single = 0;
    if (!eos_value->value.get_uint64().get(single)) {
      if (single >= *vocab_size ||
          single > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return JsonError("/eos_token_id", "token ID is outside vocabulary");
      }
      eos.push_back(static_cast<int32_t>(single));
    } else {
      auto array = artifacts::internal::Array(eos_value->value, "/eos_token_id");
      if (!array.ok()) return array.status();
      if (array->size() == 0) {
        return JsonError("/eos_token_id", "array must not be empty");
      }
      size_t index = 0;
      for (simdjson::dom::element element : *array) {
        auto value = artifacts::internal::Uint64(element, absl::StrCat("/eos_token_id/", index));
        if (!value.ok()) return value.status();
        if (*value >= *vocab_size ||
            *value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
          return JsonError(absl::StrCat("/eos_token_id/", index), "token ID is outside vocabulary");
        }
        eos.push_back(static_cast<int32_t>(*value));
        ++index;
      }
    }
  }
  std::sort(eos.begin(), eos.end());
  eos.erase(std::unique(eos.begin(), eos.end()), eos.end());

  std::optional<artifacts::ArtifactDType> dtype_hint;
  auto torch_dtype = Optional(*root, "torch_dtype");
  if (!torch_dtype.ok()) return torch_dtype.status();
  if (torch_dtype->present && !torch_dtype->value.is_null()) {
    auto value = artifacts::internal::String(torch_dtype->value, "/torch_dtype");
    if (!value.ok()) return value.status();
    if (*value == "float16" || *value == "half" || *value == "torch.float16") {
      dtype_hint = artifacts::ArtifactDType::kF16;
    } else if (*value == "bfloat16" || *value == "torch.bfloat16") {
      dtype_hint = artifacts::ArtifactDType::kBf16;
    } else if (*value == "float32" || *value == "float" || *value == "torch.float32") {
      dtype_hint = artifacts::ArtifactDType::kF32;
    } else {
      return absl::UnimplementedError("unsupported torch_dtype hint");
    }
  }

  LlamaSpec llama{*vocab_size,
                  *hidden_size,
                  *intermediate_size,
                  static_cast<uint32_t>(*layer_count),
                  static_cast<uint32_t>(*attention_heads),
                  static_cast<uint32_t>(*kv_heads),
                  static_cast<uint32_t>(*head_dim),
                  *max_positions,
                  *eps,
                  rope_theta,
                  dropout,
                  *tie,
                  *bos,
                  std::move(eos),
                  *pad,
                  dtype_hint};
  if (auto status = config.CheckUnchanged(); !status.ok()) return status;
  return ModelSpec(std::move(llama));
}

absl::StatusOr<ParameterCatalog> LlamaModelFactory::BuildParameters(const ModelSpec& spec) const {
  const LlamaSpec& llama = spec.llama();
  ParameterCatalog parameters;
  parameters.reserve(static_cast<size_t>(llama.num_hidden_layers) * 9 + 3);
  parameters.push_back(MakeParameter("model.embed_tokens.weight", ParameterRole::kTokenEmbedding,
                                     {llama.vocab_size, llama.hidden_size}));
  for (uint32_t layer = 0; layer < llama.num_hidden_layers; ++layer) {
    const std::string prefix = absl::StrCat("model.layers.", layer, ".");
    parameters.push_back(MakeParameter(prefix + "input_layernorm.weight",
                                       ParameterRole::kAttentionNorm, {llama.hidden_size}, layer));
    parameters.push_back(MakeParameter(
        prefix + "self_attn.q_proj.weight", ParameterRole::kAttentionQuery,
        {static_cast<uint64_t>(llama.num_attention_heads) * llama.head_dim, llama.hidden_size},
        layer));
    parameters.push_back(MakeParameter(
        prefix + "self_attn.k_proj.weight", ParameterRole::kAttentionKey,
        {static_cast<uint64_t>(llama.num_key_value_heads) * llama.head_dim, llama.hidden_size},
        layer));
    parameters.push_back(MakeParameter(
        prefix + "self_attn.v_proj.weight", ParameterRole::kAttentionValue,
        {static_cast<uint64_t>(llama.num_key_value_heads) * llama.head_dim, llama.hidden_size},
        layer));
    parameters.push_back(MakeParameter(
        prefix + "self_attn.o_proj.weight", ParameterRole::kAttentionOutput,
        {llama.hidden_size, static_cast<uint64_t>(llama.num_attention_heads) * llama.head_dim},
        layer));
    parameters.push_back(MakeParameter(prefix + "post_attention_layernorm.weight",
                                       ParameterRole::kMlpNorm, {llama.hidden_size}, layer));
    parameters.push_back(MakeParameter(prefix + "mlp.gate_proj.weight", ParameterRole::kMlpGate,
                                       {llama.intermediate_size, llama.hidden_size}, layer));
    parameters.push_back(MakeParameter(prefix + "mlp.up_proj.weight", ParameterRole::kMlpUp,
                                       {llama.intermediate_size, llama.hidden_size}, layer));
    parameters.push_back(MakeParameter(prefix + "mlp.down_proj.weight", ParameterRole::kMlpDown,
                                       {llama.hidden_size, llama.intermediate_size}, layer));
  }
  parameters.push_back(
      MakeParameter("model.norm.weight", ParameterRole::kFinalNorm, {llama.hidden_size}));
  parameters.push_back(MakeParameter("lm_head.weight", ParameterRole::kLmHead,
                                     {llama.vocab_size, llama.hidden_size}));
  std::sort(parameters.begin(), parameters.end(),
            [](const ParameterSpec& lhs, const ParameterSpec& rhs) {
              return lhs.canonical_name < rhs.canonical_name;
            });
  std::optional<ParameterId> embedding;
  std::optional<size_t> lm_head;
  for (size_t index = 0; index < parameters.size(); ++index) {
    parameters[index].id = ParameterId(static_cast<uint32_t>(index));
    if (parameters[index].canonical_name == "model.embed_tokens.weight") {
      embedding = parameters[index].id;
    }
    if (parameters[index].canonical_name == "lm_head.weight") lm_head = index;
  }
  if (llama.tie_word_embeddings) {
    if (!embedding.has_value() || !lm_head.has_value()) {
      return absl::InternalError("tied parameter catalog is inconsistent");
    }
    parameters[*lm_head].alias_target = *embedding;
  }
  return parameters;
}

}  // namespace inferx::model
