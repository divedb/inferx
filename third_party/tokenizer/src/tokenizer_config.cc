#include "tokenizer/config/tokenizer_config.h"

#include "tokenizer/internal/detail/json_bridge.h"
#include "tokenizer/internal/detail/loaders.h"

namespace tokenizer {

StatusOr<TokenizerConfig> ParseTokenizerConfig(std::string_view text) {
  ABSL_ASSIGN_OR_RETURN(Json parsed, ParseJson(text, "tokenizer_config.json"));
  if (!parsed.is_object()) {
    return InvalidArgumentError("tokenizer_config.json must be a JSON object");
  }

  TokenizerConfig config;
  config.tokenizer_class = GetString(parsed, "tokenizer_class").value_or("");

  // Hugging Face writes an enormous float here for "no limit" -- gpt-oss-20b
  // ships one that does not fit in int64_t -- so this clamps rather than
  // wrapping into a small, wrong maximum.
  config.model_max_length =
      GetClampedInt(parsed, "model_max_length", kNoMaxLength).value_or(kNoMaxLength);

  config.add_bos_token = GetBool(parsed, "add_bos_token");
  config.add_eos_token = GetBool(parsed, "add_eos_token");
  config.clean_up_tokenization_spaces = GetBool(parsed, "clean_up_tokenization_spaces");
  config.add_prefix_space = GetBool(parsed, "add_prefix_space");
  config.split_special_tokens = GetBool(parsed, "split_special_tokens");

  if (auto side = GetString(parsed, "padding_side")) {
    config.padding_side = *side;
  }
  if (auto side = GetString(parsed, "truncation_side")) {
    config.truncation_side = *side;
  }
  config.model_input_names = GetStringArray(parsed, "model_input_names");

  config.raw = JsonBridge::ToPublic(parsed);
  return config;
}

}  // namespace tokenizer
