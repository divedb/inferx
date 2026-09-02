#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tokenizer/core/json_value.h"
#include "tokenizer/core/types.h"

namespace tokenizer {

/// \brief A typed view of `tokenizer_config.json`.
///
/// Every field is optional in the file, so every field here either is an
/// `optional` or carries the documented Hugging Face default. `gpt2` ships a
/// config containing only `model_max_length` and must load cleanly.
struct TokenizerConfig {
  /// The Python class the checkpoint was saved from -- "Qwen2Tokenizer",
  /// "LlamaTokenizerFast", ... This is what adapters dispatch on. Never a
  /// model name.
  std::string tokenizer_class;

  int64_t model_max_length = kNoMaxLength;

  /// Llama-family flags. The Rust post-processor does not reflect them (see
  /// LlamaFamilyAdapter), so they are applied above the backend.
  std::optional<bool> add_bos_token;
  std::optional<bool> add_eos_token;

  /// Applied on decode. Lives in Python upstream, so the Rust decoder never
  /// does it. `nullopt` means the config omitted it, which Hugging Face
  /// historically treated as `true`.
  std::optional<bool> clean_up_tokenization_spaces;

  std::optional<bool> add_prefix_space;
  std::optional<bool> split_special_tokens;

  std::string padding_side = "right";
  std::string truncation_side = "right";

  std::vector<std::string> model_input_names;

  /// The whole file, verbatim, for adapters and forward compatibility.
  JsonValue raw;
};

}  // namespace tokenizer
