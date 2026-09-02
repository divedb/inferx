// The one translation unit allowed to include the vendored Jinja engine.
// Nothing under include/ may reach it, and no other .cc needs to: this file
// is the whole binding.

#include "tokenizer/chat/chat_template.h"

#include <exception>
#include <minja/chat-template.hpp>
#include <utility>

#include "absl/time/clock.h"
#include "tokenizer/internal/chat/chat_message_json.h"
#include "tokenizer/internal/detail/json_bridge.h"

namespace tokenizer {
namespace {

ChatTemplateCaps ToCaps(const minja::chat_template_caps& caps) {
  ChatTemplateCaps out;
  out.supports_tools = caps.supports_tools;
  out.supports_tool_calls = caps.supports_tool_calls;
  out.supports_tool_responses = caps.supports_tool_responses;
  out.supports_system_role = caps.supports_system_role;
  out.supports_parallel_tool_calls = caps.supports_parallel_tool_calls;
  out.supports_tool_call_id = caps.supports_tool_call_id;
  out.requires_object_arguments = caps.requires_object_arguments;
  out.requires_non_null_content = caps.requires_non_null_content;
  out.requires_typed_content = caps.requires_typed_content;
  return out;
}

}  // namespace

struct ChatTemplate::Impl {
  std::unique_ptr<minja::chat_template> tmpl;
  ChatTemplateCaps caps;
  std::string source;
};

ChatTemplate::ChatTemplate() : impl_(std::make_unique<Impl>()) {}
ChatTemplate::ChatTemplate(ChatTemplate&&) noexcept = default;
ChatTemplate& ChatTemplate::operator=(ChatTemplate&&) noexcept = default;
ChatTemplate::~ChatTemplate() = default;

StatusOr<ChatTemplate> ChatTemplate::Create(std::string source, std::string bos_token,
                                            std::string eos_token) {
  if (source.empty()) {
    return InvalidArgumentError("the chat template source is empty");
  }

  ChatTemplate out;
  try {
    // Construction compiles the template and probes its capabilities by
    // rendering it with synthetic messages, so this is where a malformed
    // template surfaces.
    out.impl_->tmpl = std::make_unique<minja::chat_template>(source, bos_token, eos_token);
  } catch (const std::exception& error) {
    return InvalidArgumentError("cannot compile the chat template: ", error.what());
  }
  out.impl_->caps = ToCaps(out.impl_->tmpl->original_caps());
  out.impl_->source = std::move(source);
  return out;
}

StatusOr<std::string> ChatTemplate::Apply(absl::Span<const ChatMessage> messages,
                                          const ChatTemplateOptions& options) const {
  if (impl_->tmpl == nullptr) {
    return FailedPreconditionError("the chat template is not compiled");
  }

  minja::chat_template_inputs inputs;
  inputs.messages = MessagesToJson(messages);
  inputs.add_generation_prompt = options.add_generation_prompt;

  if (!options.tools.empty()) {
    OrderedJson tools = OrderedJson::array();
    for (const JsonValue& tool : options.tools) {
      tools.push_back(JsonBridge::ToOrdered(tool));
    }
    inputs.tools = std::move(tools);
  }

  if (!options.extra_context.empty()) {
    OrderedJson extra = OrderedJson::object();
    for (const auto& [key, value] : options.extra_context) {
      extra[key] = JsonBridge::ToOrdered(value);
    }
    inputs.extra_context = std::move(extra);
  }

  // A template calling strftime_now is otherwise non-deterministic, which
  // makes its output impossible to test. Callers pin the clock; the engine
  // takes the timestamp directly.
  if (options.now.has_value()) {
    inputs.now = absl::ToChronoTime(*options.now);
  }

  minja::chat_template_options engine_options;
  engine_options.apply_polyfills = options.apply_polyfills;

  try {
    return impl_->tmpl->apply(inputs, engine_options);
  } catch (const std::exception& error) {
    // Templates signal unsupported conversations by raising. The message is
    // the template author's and is worth passing through verbatim.
    Status status = InvalidArgumentError("chat template failed to render: ", error.what());
    status.SetPayload("type.tokenizer.dev/reason", absl::Cord("TEMPLATE_RENDER"));
    return status;
  }
}

const ChatTemplateCaps& ChatTemplate::caps() const { return impl_->caps; }

std::string_view ChatTemplate::source() const { return impl_->source; }

}  // namespace tokenizer
