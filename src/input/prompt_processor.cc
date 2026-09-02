// Prompt processing (ADR 0025).

#include "inferx/input/prompt_processor.h"

#include <algorithm>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"

namespace inferx::input {
namespace {

absl::Status InvalidPrompt(std::string_view message) {
  return absl::InvalidArgumentError(absl::StrCat("prompt: ", message));
}

bool ValidUtf8(std::string_view text) {
  const auto* p = reinterpret_cast<const unsigned char*>(text.data());
  const auto* end = p + text.size();
  while (p < end) {
    const unsigned char c = *p;
    size_t extra;
    uint32_t code;
    if (c < 0x80) {
      ++p;
      continue;
    }
    if ((c & 0xE0) == 0xC0) {
      extra = 1;
      code = c & 0x1F;
    } else if ((c & 0xF0) == 0xE0) {
      extra = 2;
      code = c & 0x0F;
    } else if ((c & 0xF8) == 0xF0) {
      extra = 3;
      code = c & 0x07;
    } else {
      return false;
    }
    if (p + extra >= end) return false;
    for (size_t i = 1; i <= extra; ++i) {
      const unsigned char cc = p[i];
      if ((cc & 0xC0) != 0x80) return false;
      code = (code << 6) | (cc & 0x3F);
    }
    if (extra == 1 && code < 0x80) return false;
    if (extra == 2 && code < 0x800) return false;
    if (extra == 3 && code < 0x10000) return false;
    if (code > 0x10FFFF) return false;
    if (code >= 0xD800 && code <= 0xDFFF) return false;
    p += extra + 1;
  }
  return true;
}

}  // namespace

absl::StatusOr<GenerateRequest> PromptProcessor::Process(const InputProcessingRequest& request,
                                                         const PromptModelFacts& facts,
                                                         const PromptLimits& limits,
                                                         MonotonicTime now) {
  if (facts.tokenizer == nullptr) {
    return absl::FailedPreconditionError("prompt: no tokenizer is loaded");
  }
  if (request.deadline.has_value() && now >= *request.deadline) {
    return absl::DeadlineExceededError("prompt: deadline already expired");
  }
  if (request.max_output_tokens.value() == 0) {
    return InvalidPrompt("maximum output tokens must be positive");
  }

  std::vector<TokenId> prompt_tokens;
  std::string rendered;  // chat only; discarded before return

  if (const auto* raw = std::get_if<RawTextPrompt>(&request.prompt)) {
    if (raw->text.size() > limits.max_input_bytes) {
      return absl::ResourceExhaustedError(absl::StrCat("prompt: ", raw->text.size(),
                                                       " bytes exceeds the per-request "
                                                       "input limit of ",
                                                       limits.max_input_bytes));
    }
    if (!ValidUtf8(raw->text)) {
      return InvalidPrompt("input is not valid UTF-8");
    }
    tokenization::EncodeOptions options;
    options.add_special_tokens = raw->add_special_tokens;
    auto encoded = facts.tokenizer->Encode(raw->text, options);
    if (!encoded.ok()) {
      return absl::Status(encoded.status().code(),
                          absl::StrCat("prompt: ", encoded.status().message()));
    }
    prompt_tokens = std::move(*encoded);
  } else if (const auto* chat = std::get_if<ChatPrompt>(&request.prompt)) {
    if (chat->messages.empty()) {
      return InvalidPrompt("chat prompt needs at least one message");
    }
    uint64_t total_bytes = 0;
    std::vector<tokenization::ChatMessage> messages;
    messages.reserve(chat->messages.size());
    for (const ChatMessage& message : chat->messages) {
      if (message.content.size() > limits.max_input_bytes) {
        return absl::ResourceExhaustedError(
            absl::StrCat("prompt: message of ", message.content.size(),
                         " bytes exceeds the per-request input limit"));
      }
      if (!ValidUtf8(message.content)) {
        return InvalidPrompt("chat message is not valid UTF-8");
      }
      total_bytes += message.content.size();
      if (total_bytes > limits.max_input_bytes) {
        return absl::ResourceExhaustedError(
            "prompt: aggregate chat bytes exceed the per-request limit");
      }
      tokenization::ChatMessage converted;
      converted.role = message.role == ChatMessage::Role::kSystem
                           ? tokenization::ChatMessage::Role::kSystem
                           : (message.role == ChatMessage::Role::kAssistant
                                  ? tokenization::ChatMessage::Role::kAssistant
                                  : tokenization::ChatMessage::Role::kUser);
      converted.content = message.content;
      messages.push_back(std::move(converted));
    }
    tokenization::ChatRenderOptions render;
    render.add_generation_prompt = chat->add_generation_prompt;
    render.template_name = chat->template_name;
    auto rendered_result = facts.tokenizer->RenderChatPrompt(messages, render);
    if (!rendered_result.ok()) {
      return absl::Status(rendered_result.status().code(),
                          absl::StrCat("prompt: ", rendered_result.status().message()));
    }
    rendered = std::move(*rendered_result);
    if (rendered.size() > limits.max_rendered_bytes) {
      return absl::ResourceExhaustedError(absl::StrCat("prompt: rendered template is ",
                                                       rendered.size(),
                                                       " bytes and exceeds the rendered limit"));
    }
    if (!ValidUtf8(rendered)) {
      return InvalidPrompt("rendered chat template is not valid UTF-8");
    }
    // The template already emitted the checkpoint's special tokens; encoding
    // must not add them again (double-BOS).
    tokenization::EncodeOptions options;
    options.add_special_tokens = false;
    auto encoded = facts.tokenizer->Encode(rendered, options);
    if (!encoded.ok()) {
      return absl::Status(encoded.status().code(),
                          absl::StrCat("prompt: ", encoded.status().message()));
    }
    prompt_tokens = std::move(*encoded);
  } else {
    const auto& tokens = std::get<TokenIdPrompt>(request.prompt).tokens;
    prompt_tokens = tokens;
  }

  // Post-encode validation for every variant.
  if (prompt_tokens.empty()) {
    return InvalidPrompt("tokenizes to an empty prompt");
  }
  if (prompt_tokens.size() > limits.max_prompt_tokens) {
    return absl::ResourceExhaustedError(
        absl::StrCat("prompt: ", prompt_tokens.size(), " tokens exceeds the prompt token limit"));
  }
  for (TokenId id : prompt_tokens) {
    if (id.value() < 0) {
      return InvalidPrompt("negative token id in prompt");
    }
    if (static_cast<uint64_t>(id.value()) >= facts.model_vocab_size) {
      return absl::InvalidArgumentError(absl::StrCat("prompt: token id ", id.value(),
                                                     " is outside the model vocabulary of ",
                                                     facts.model_vocab_size));
    }
  }
  const uint64_t total = static_cast<uint64_t>(prompt_tokens.size()) +
                         static_cast<uint64_t>(request.max_output_tokens.value());
  if (total > facts.max_context_tokens.value()) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "prompt: ", prompt_tokens.size(), " prompt + ", request.max_output_tokens.value(),
        " output tokens exceeds the effective context of ", facts.max_context_tokens.value()));
  }

  GenerateRequest out{request.request_id,       request.model_id,
                      std::move(prompt_tokens), GenerationLimits{request.max_output_tokens},
                      kDefaultPriority,         request.deadline};
  return out;
}

}  // namespace inferx::input
