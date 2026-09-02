// Adapted from divedb/tokenizer (MIT) for InferX local-only use (ADR 0024):
// ModelDir's filesystem discovery is replaced by caller-provided bytes, so
// every filesystem decision stays in the host's rooted artifact layer.

#pragma once

// INTERNAL. Loading-path input values.

#include <optional>
#include <string>
#include <utility>

namespace tokenizer {

/// \brief The checkpoint artifacts the loader consumes, as bytes.
///
/// The host owns discovery: it reads these files through its own rooted,
/// validated session and hands the bytes over. This package performs no
/// filesystem or network access anywhere.
struct LocalArtifacts {
  /// Serialized `tokenizer.json`. Required.
  std::string tokenizer_json;

  /// Optional checkpoint metadata, in increasing order of use.
  std::optional<std::string> tokenizer_config;     // tokenizer_config.json
  std::optional<std::string> special_tokens_map;   // special_tokens_map.json
  std::optional<std::string> added_tokens;         // added_tokens.json
  std::optional<std::string> chat_template_jinja;  // chat_template.jinja
  std::optional<std::string> chat_template_json;   // chat_template.json
  std::optional<std::string> config;               // config.json
  std::optional<std::string> generation_config;    // generation_config.json
};

}  // namespace tokenizer
