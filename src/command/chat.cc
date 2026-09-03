#include "common.h"

namespace inferx::command {

void Chat(const ChatOptions&) {
  internal::Unavailable("chat", "the OpenAI-compatible HTTP client is not compiled in this milestone");
}

}  // namespace inferx::command
