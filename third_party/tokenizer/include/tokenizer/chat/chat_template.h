#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tokenizer/chat/chat_message.h"
#include "tokenizer/core/json_value.h"
#include "tokenizer/core/status.h"

namespace tokenizer {

/// \brief What a checkpoint's template actually supports.
///
/// Probed by rendering the template with synthetic messages -- a real feature
/// of the vendored Jinja engine, surfaced here rather than reimplemented.
/// Useful to a server deciding whether it can offer tool calling at all.
struct ChatTemplateCaps {
  bool supports_tools = false;
  bool supports_tool_calls = false;
  bool supports_tool_responses = false;
  bool supports_system_role = false;
  bool supports_parallel_tool_calls = false;
  bool supports_tool_call_id = false;
  bool requires_object_arguments = false;
  bool requires_non_null_content = false;
  bool requires_typed_content = false;
};

struct ChatTemplateOptions {
  /// Append the assistant header so the model continues as the assistant
  /// rather than predicting the next speaker.
  bool add_generation_prompt = false;

  /// Selects among a checkpoint's named templates. A checkpoint may ship
  /// several (e.g. a "tool_use" variant alongside the default).
  std::string template_name = "default";

  /// JSON-schema tool definitions, in the shape the template expects.
  std::vector<JsonValue> tools;

  /// Extra template variables: `enable_thinking` for Qwen3, and anything else
  /// a checkpoint's template reads.
  absl::flat_hash_map<std::string, JsonValue> extra_context;

  /// Pins the clock the template sees. Templates calling `strftime_now` are
  /// otherwise non-deterministic, which makes their output untestable.
  std::optional<absl::Time> now;

  /// Enable the engine's polyfills for templates that lack a capability
  /// (a missing system role, tool calls the template cannot express). Off
  /// gives exactly what the checkpoint's template produces, which is what the
  /// Hugging Face compatibility tests compare against.
  bool apply_polyfills = false;
};

/// \brief A checkpoint's chat template, ready to render.
///
/// A thin owner over the vendored Jinja engine: it holds the compiled
/// template and converts messages into the shape the engine wants. Everything
/// about Jinja -- parsing, evaluation, `raise_exception`, `strftime_now`,
/// `tojson`, capability probing, polyfills -- belongs to the engine and is not
/// reimplemented here.
///
/// There is deliberately no interface to implement: one engine, one
/// implementation. The pimpl exists to keep the engine's headers (and the
/// JSON type in its signatures) out of this one.
class ChatTemplate {
 public:
  /// \brief Compiles `source`.
  ///
  /// `bos_token` and `eos_token` are bound as template variables; templates
  /// reference them directly. Construction runs capability probing, so it is
  /// not free -- callers should build lazily.
  static StatusOr<ChatTemplate> Create(std::string source, std::string bos_token,
                                       std::string eos_token);

  ChatTemplate(ChatTemplate&&) noexcept;
  ChatTemplate& operator=(ChatTemplate&&) noexcept;
  ChatTemplate(const ChatTemplate&) = delete;
  ChatTemplate& operator=(const ChatTemplate&) = delete;
  ~ChatTemplate();

  /// \brief Renders `messages`.
  ///
  /// Const, and the engine holds no state across renders, so this is safe to
  /// call concurrently on one instance -- unlike the tokenizer that owns it.
  ///
  /// \returns The prompt text; InvalidArgument if the template raised or
  ///          failed to render, carrying the template's own message.
  StatusOr<std::string> Apply(absl::Span<const ChatMessage> messages,
                              const ChatTemplateOptions& options) const;

  const ChatTemplateCaps& caps() const;
  std::string_view source() const;

 private:
  ChatTemplate();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tokenizer
