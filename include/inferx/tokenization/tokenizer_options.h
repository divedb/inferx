// Tokenizer call options. The serving default for
// decode is cleanup disabled: cleanup can rewrite already-emitted bytes and
// is not part of the streaming contract.

#ifndef INFERX_TOKENIZATION_TOKENIZER_OPTIONS_H_
#define INFERX_TOKENIZATION_TOKENIZER_OPTIONS_H_

#include <cstdint>
#include <optional>

#include "inferx/base/token.h"

namespace inferx::tokenization {

enum class TruncationSide : uint8_t {
  kRight,
  kLeft,
};

struct EncodeOptions {
  bool add_special_tokens = true;
  // Applied after special tokens are in place; a plain slice, exactly like
  // Hugging Face's truncation without padding.
  std::optional<TokenCount> max_tokens;
  TruncationSide truncation_side = TruncationSide::kRight;
};

struct DecodeOptions {
  bool skip_special_tokens = false;
  // The streaming decoder never applies cleanup.
  bool clean_up_tokenization_spaces = false;
};

}  // namespace inferx::tokenization

#endif  // INFERX_TOKENIZATION_TOKENIZER_OPTIONS_H_
