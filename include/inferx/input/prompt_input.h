// Input-variant values for prompt processing. Exactly
// one representation per request; the variant is closed.

#ifndef INFERX_INPUT_PROMPT_INPUT_H_
#define INFERX_INPUT_PROMPT_INPUT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "inferx/api/generate_request.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"

namespace inferx::input {

struct RawTextPrompt {
  // Owned bytes; validated as UTF-8 and against the byte limit by the
  // processor, not by construction.
  std::string text;
  // Explicit add-special-tokens policy; default true for raw text.
  bool add_special_tokens = true;
};

struct ChatMessage {
  enum class Role : uint8_t { kSystem, kUser, kAssistant };
  Role role = Role::kUser;
  std::string content;  // valid UTF-8, validated by the processor
};

struct ChatPrompt {
  std::vector<ChatMessage> messages;
  bool add_generation_prompt = true;
  // Only an explicitly named template or the checkpoint-declared default is
  // used; there is no fallback template.
  std::optional<std::string> template_name;
};

struct TokenIdPrompt {
  std::vector<TokenId> tokens;  // validated against the model vocabulary
};

using PromptInput = std::variant<RawTextPrompt, ChatPrompt, TokenIdPrompt>;

// The processing request crossing into the input layer. Mirrors the
// GenerateRequest fields that validation needs, without text.
struct InputProcessingRequest {
  RequestId request_id;
  RequestEpoch epoch;
  ModelId model_id;
  PromptInput prompt;
  // Maximum output tokens for the context-fit check.
  TokenCount max_output_tokens;
  std::optional<MonotonicTime> deadline;
};

}  // namespace inferx::input

#endif  // INFERX_INPUT_PROMPT_INPUT_H_
