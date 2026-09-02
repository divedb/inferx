#include "tokenizer/internal/adapters/llama_family_adapter.h"

#include "tokenizer/tokenization/pretrained_tokenizer.h"

namespace tokenizer {

std::string_view LlamaFamilyAdapter::name() const { return "llama_family"; }

void LlamaFamilyAdapter::PostEncode(std::vector<TokenId>& ids, const EncodeOptions& options,
                                    const TokenizerConfig& config,
                                    const SpecialTokens& special) const {
  if (!options.add_special_tokens) return;

  if (config.add_bos_token.value_or(false)) {
    const std::optional<TokenId> bos = special.bos_id();
    // Guarded rather than unconditional: a checkpoint whose serialized
    // post-processor *does* add BOS would otherwise get it twice.
    if (bos.has_value() && (ids.empty() || ids.front() != *bos)) {
      ids.insert(ids.begin(), *bos);
    }
  }

  if (config.add_eos_token.value_or(false)) {
    const std::optional<TokenId> eos = special.eos_id();
    if (eos.has_value() && (ids.empty() || ids.back() != *eos)) {
      ids.push_back(*eos);
    }
  }
}

}  // namespace tokenizer
