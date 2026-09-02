// Adapted from divedb/tokenizer (MIT) for InferX local-only use (ADR 0024):
// every Hub/cache/network option is removed; loading consumes bytes through
// LocalArtifacts only.

#pragma once

#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"

namespace tokenizer {

struct PretrainedTokenizerOptions {
  /// Forces a specific adapter/tokenizer class instead of reading
  /// `tokenizer_class` from `tokenizer_config.json`.
  std::optional<std::string> tokenizer_type;

  /// Promotes load-time cross-check warnings (an EOS token missing from the
  /// vocabulary, a config.json id disagreeing with the resolved one) into
  /// errors.
  bool strict = false;

  /// Supplies a chat template for a checkpoint that ships none. Opt-in; a
  /// template is never inferred from a model name.
  std::optional<std::string> chat_template_override;

  /// Overrides resolved special tokens by role name ("bos_token",
  /// "eos_token", ...). Highest precedence of all sources.
  absl::flat_hash_map<std::string, std::string> special_token_overrides;
};

}  // namespace tokenizer
