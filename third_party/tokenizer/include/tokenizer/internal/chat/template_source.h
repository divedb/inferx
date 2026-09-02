// Adapted from divedb/tokenizer (MIT) for InferX local-only use (ADR 0024):
// template discovery consumes LocalArtifacts bytes instead of directory
// scans; the additional_chat_templates/*.jinja side directory is dropped
// because InferX opens only fixed well-known checkpoint files.

#pragma once

// INTERNAL.

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tokenizer/core/status.h"
#include "tokenizer/internal/detail/json_bridge.h"
#include "tokenizer/internal/detail/local_artifacts.h"

namespace tokenizer {

/// \brief The chat templates a checkpoint ships, as source text.
///
/// Discovery is ours; rendering is the engine's. Sources are collected at
/// load and compiled on first use, because compiling probes a template's
/// capabilities by rendering it several times -- real work that a caller who
/// only encodes text should not pay for.
struct ChatTemplateSources {
  absl::flat_hash_map<std::string, std::string> by_name;
  std::string default_name = "default";

  bool empty() const { return by_name.empty(); }
  const std::string* Find(std::string_view name) const;
  std::vector<std::string> Names() const;
};

/// \brief Collects templates from a checkpoint, first source wins.
///
/// The order mirrors modern Transformers:
///   1. `chat_template.jinja`   -- canonical since 4.51 (gpt-oss ships this);
///   2. `tokenizer_config.json["chat_template"]` as a string -- the common
///      case (Qwen, DeepSeek);
///   3. the same key as a list of {name, template} -- the deprecated
///      multi-template form;
///   4. `chat_template.json`    -- processor-level, some multimodal repos.
///
/// `tokenizer_config` may be a null JSON value when the checkpoint ships no
/// `tokenizer_config.json`.
ChatTemplateSources DiscoverChatTemplates(const LocalArtifacts& artifacts,
                                          const Json& tokenizer_config);

}  // namespace tokenizer
