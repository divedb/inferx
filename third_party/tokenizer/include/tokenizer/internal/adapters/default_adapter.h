#pragma once

#include <string>
#include <string_view>

#include "tokenizer/adapters/adapter.h"

namespace tokenizer {

/// \brief The adapter every checkpoint gets unless a family registers its own.
///
/// Implements only behaviour that is driven by configuration and shared by
/// all families: currently `clean_up_tokenization_spaces`, which Hugging Face
/// applies in Python and the Rust decoder therefore never does.
class DefaultAdapter : public TokenizerAdapter {
 public:
  std::string_view name() const override;

  void PostDecode(std::string& text, const DecodeOptions& options,
                  const TokenizerConfig& config) const override;
};

}  // namespace tokenizer
