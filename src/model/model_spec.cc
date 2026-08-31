#include "inferx/model/model_spec.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace inferx::model {
namespace {

template <typename Unsigned>
void AppendUnsigned(std::vector<std::byte>& output, Unsigned value) {
  static_assert(std::is_unsigned_v<Unsigned>);
  for (size_t i = 0; i < sizeof(value); ++i) {
    output.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFFU));
  }
}

void AppendOptionalToken(std::vector<std::byte>& output, std::optional<int32_t> token) {
  output.push_back(token.has_value() ? std::byte{1} : std::byte{0});
  if (token.has_value()) {
    AppendUnsigned(output, static_cast<uint32_t>(*token));
  }
}

std::vector<std::byte> Canonicalize(const LlamaSpec& spec) {
  std::vector<std::byte> output;
  output.reserve(160 + spec.eos_token_ids.size() * sizeof(uint32_t));
  AppendUnsigned(output, ModelSpec::kSchemaVersion);
  AppendUnsigned(output, spec.vocab_size);
  AppendUnsigned(output, spec.hidden_size);
  AppendUnsigned(output, spec.intermediate_size);
  AppendUnsigned(output, spec.num_hidden_layers);
  AppendUnsigned(output, spec.num_attention_heads);
  AppendUnsigned(output, spec.num_key_value_heads);
  AppendUnsigned(output, spec.head_dim);
  AppendUnsigned(output, spec.max_position_embeddings);
  const double normalized_eps = spec.rms_norm_eps == 0 ? 0 : spec.rms_norm_eps;
  const double normalized_theta = spec.rope_theta == 0 ? 0 : spec.rope_theta;
  const double normalized_dropout = spec.attention_dropout == 0 ? 0 : spec.attention_dropout;
  AppendUnsigned(output, std::bit_cast<uint64_t>(normalized_eps));
  AppendUnsigned(output, std::bit_cast<uint64_t>(normalized_theta));
  AppendUnsigned(output, std::bit_cast<uint64_t>(normalized_dropout));
  output.push_back(spec.tie_word_embeddings ? std::byte{1} : std::byte{0});
  AppendOptionalToken(output, spec.bos_token_id);
  AppendUnsigned(output, static_cast<uint64_t>(spec.eos_token_ids.size()));
  for (const int32_t token : spec.eos_token_ids) {
    AppendUnsigned(output, static_cast<uint32_t>(token));
  }
  AppendOptionalToken(output, spec.pad_token_id);
  output.push_back(spec.source_weight_type_hint.has_value() ? std::byte{1} : std::byte{0});
  if (spec.source_weight_type_hint.has_value()) {
    AppendUnsigned(output, static_cast<uint32_t>(*spec.source_weight_type_hint));
  }
  return output;
}

}  // namespace

ModelSpec::ModelSpec(LlamaSpec llama)
    : llama_(std::move(llama)), canonical_(Canonicalize(llama_)) {}

}  // namespace inferx::model
