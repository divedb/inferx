#include "tokenizer/internal/chat/chat_message_json.h"

namespace tokenizer {
namespace {

OrderedJson ToolCallToJson(const ToolCall& call) {
  OrderedJson function = OrderedJson::object();
  function["name"] = call.name;

  // Arguments travel as text because templates disagree about whether they
  // want an object or a stringified one. Parse when it is valid JSON so a
  // template expecting an object gets one; the engine's own polyfill handles
  // the reverse case.
  OrderedJson arguments = OrderedJson::parse(call.arguments, nullptr, /*allow_exceptions=*/false);
  if (arguments.is_discarded()) {
    function["arguments"] = call.arguments;
  } else {
    function["arguments"] = std::move(arguments);
  }

  OrderedJson out = OrderedJson::object();
  if (!call.id.empty()) out["id"] = call.id;
  out["type"] = call.type.empty() ? std::string("function") : call.type;
  out["function"] = std::move(function);
  return out;
}

OrderedJson ContentPartToJson(const ContentPart& part) {
  OrderedJson out = OrderedJson::object();
  out["type"] = part.type;
  if (part.type == "text") out["text"] = part.text;
  if (!part.extra.IsNull()) {
    OrderedJson extra = JsonBridge::ToOrdered(part.extra);
    if (extra.is_object()) {
      for (auto it = extra.begin(); it != extra.end(); ++it) {
        out[it.key()] = it.value();
      }
    }
  }
  return out;
}

}  // namespace

OrderedJson MessagesToJson(absl::Span<const ChatMessage> messages) {
  OrderedJson out = OrderedJson::array();
  for (const ChatMessage& message : messages) {
    OrderedJson entry = OrderedJson::object();
    entry["role"] = message.role;

    if (!message.content_parts.empty()) {
      OrderedJson parts = OrderedJson::array();
      for (const ContentPart& part : message.content_parts) {
        parts.push_back(ContentPartToJson(part));
      }
      entry["content"] = std::move(parts);
    } else {
      entry["content"] = message.content;
    }

    if (message.reasoning_content.has_value()) {
      entry["reasoning_content"] = *message.reasoning_content;
    }
    if (message.name.has_value()) entry["name"] = *message.name;
    if (message.tool_call_id.has_value()) {
      entry["tool_call_id"] = *message.tool_call_id;
    }
    if (!message.tool_calls.empty()) {
      OrderedJson calls = OrderedJson::array();
      for (const ToolCall& call : message.tool_calls) {
        calls.push_back(ToolCallToJson(call));
      }
      entry["tool_calls"] = std::move(calls);
    }
    for (const auto& [key, value] : message.extra) {
      entry[key] = JsonBridge::ToOrdered(value);
    }

    out.push_back(std::move(entry));
  }
  return out;
}

}  // namespace tokenizer
