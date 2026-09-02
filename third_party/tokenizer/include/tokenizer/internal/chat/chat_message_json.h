#pragma once

// INTERNAL.

#include "absl/types/span.h"
#include "tokenizer/chat/chat_message.h"
#include "tokenizer/internal/detail/json_bridge.h"

namespace tokenizer {

/// \brief Converts messages into the JSON shape Hugging Face templates expect.
///
/// Ordered, because template output is compared byte-for-byte against the
/// reference implementation and `tojson` of a message would otherwise emit
/// keys in a different order.
///
/// Empty optional fields are omitted rather than written as null: templates
/// routinely branch on `{% if message.tool_calls is defined %}`, and a null
/// would take the wrong branch.
OrderedJson MessagesToJson(absl::Span<const ChatMessage> messages);

}  // namespace tokenizer
