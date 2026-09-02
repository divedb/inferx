#pragma once

// INTERNAL.

#include <string>
#include <string_view>

#include "tokenizer/internal/adapters/default_adapter.h"

namespace tokenizer {

/// \brief Restores the BOS/EOS insertion Hugging Face performs in Python.
///
/// `LlamaTokenizerFast.__init__` rewrites the serialized post-processor from
/// `add_bos_token` / `add_eos_token` at construction time. The Rust engine
/// only ever sees the *serialized* post-processor, which does not reflect
/// that, so `add_special_tokens=true` alone is not enough to match the
/// reference.
///
/// DeepSeek-V2-Lite is exactly this case: it declares `add_bos_token: true`
/// while its `tokenizer.json` carries a plain ByteLevel post-processor.
/// Without this adapter every prompt is silently missing its BOS token.
///
/// Rewriting the serialized post-processor before handing it to the engine
/// would be closer to what Hugging Face does, but it means generating the
/// engine's internal JSON, whose schema changes between engine versions.
/// Adjusting the ids afterwards is a few lines and survives upgrades.
class LlamaFamilyAdapter : public DefaultAdapter {
 public:
  std::string_view name() const override;

  void PostEncode(std::vector<TokenId>& ids, const EncodeOptions& options,
                  const TokenizerConfig& config,
                  const SpecialTokens& special_tokens) const override;
};

}  // namespace tokenizer
