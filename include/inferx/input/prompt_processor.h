// Prompt processing (ADR 0025): exactly one input
// variant, template-then-encode-once, no inferred template, limits after
// tokenization. The processor returns a GenerateRequest so scheduling
// never receives text or dependency objects.

#ifndef INFERX_INPUT_PROMPT_PROCESSOR_H_
#define INFERX_INPUT_PROMPT_PROCESSOR_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "inferx/api/generate_request.h"
#include "inferx/base/clock.h"
#include "inferx/input/prompt_input.h"
#include "inferx/tokenization/tokenizer.h"

namespace inferx::input {

// Byte/token limits applied by the processor.
struct PromptLimits {
  // Per request before template rendering.
  uint64_t max_input_bytes = 4u << 20;
  // Checked while rendering, before encode.
  uint64_t max_rendered_bytes = 8u << 20;
  // Maximum prompt tokens admitted per request.
  uint64_t max_prompt_tokens = 32768;
};

// What the processor needs to know about the target model. Values come from
// the validated package; the processor performs the cross-cutting checks.
struct PromptModelFacts {
  const tokenization::Tokenizer* tokenizer = nullptr;
  // ModelSpec vocab size: every emitted id must be below it.
  uint64_t model_vocab_size = 0;
  // Effective request cap: min(server, model execution) context.
  TokenCount max_context_tokens{0};
};

class PromptProcessor {
 public:
  // Validates the prompt, renders and encodes it exactly once, and returns
  // a GenerateRequest carrying owned token ids.
  static absl::StatusOr<GenerateRequest> Process(const InputProcessingRequest& request,
                                                 const PromptModelFacts& facts,
                                                 const PromptLimits& limits, MonotonicTime now);
};

}  // namespace inferx::input

#endif  // INFERX_INPUT_PROMPT_PROCESSOR_H_
