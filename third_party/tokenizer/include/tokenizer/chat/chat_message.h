#pragma once

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tokenizer/core/json_value.h"

namespace tokenizer {

/// \brief One element of a typed `content` array (multimodal templates).
struct ContentPart {
  std::string type;  ///< "text", "image", "image_url", ...
  std::string text;  ///< Populated when `type == "text"`.
  JsonValue extra;   ///< Everything else, verbatim.
};

/// \brief An assistant's request to call a tool.
struct ToolCall {
  std::string id;
  std::string type = "function";
  std::string name;

  /// JSON-encoded arguments. Kept as text because templates disagree about
  /// whether they want an object or a stringified object, and minja's
  /// capability probing polyfills whichever the template expects.
  std::string arguments;
};

/// \brief One turn of a conversation.
///
/// `role` is a free-form string, deliberately not an enum: gpt-oss uses
/// "developer", Llama 3.1 uses "ipython", and a new model family should not
/// require a header change to serve.
struct ChatMessage {
  std::string role;
  std::string content;

  /// Set instead of `content` when the template expects typed parts.
  std::vector<ContentPart> content_parts;

  std::vector<ToolCall> tool_calls;
  std::optional<std::string> tool_call_id;  ///< For role == "tool".
  std::optional<std::string> name;

  /// Qwen3 / DeepSeek-R1 style separated reasoning.
  std::optional<std::string> reasoning_content;

  /// Forward compatibility: anything a future template reads that this struct
  /// does not name. Merged into the message object as-is.
  absl::flat_hash_map<std::string, JsonValue> extra;

  ChatMessage() = default;
  ChatMessage(std::string role_in, std::string content_in)
      : role(std::move(role_in)), content(std::move(content_in)) {}
};

}  // namespace tokenizer
