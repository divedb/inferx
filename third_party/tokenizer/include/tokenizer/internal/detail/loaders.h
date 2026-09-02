// Adapted from divedb/tokenizer (MIT) for InferX local-only use (ADR 0024):
// the ModelDir/filesystem signatures are replaced by LocalArtifacts bytes.

#pragma once

// INTERNAL. Free functions shared by the loading path.

#include <string>
#include <string_view>
#include <vector>

#include "tokenizer/config/pretrained_options.h"
#include "tokenizer/config/tokenizer_config.h"
#include "tokenizer/core/status.h"
#include "tokenizer/internal/backend/backend.h"
#include "tokenizer/internal/detail/local_artifacts.h"
#include "tokenizer/tokenization/special_tokens.h"

namespace tokenizer {

/// \brief Parses `tokenizer_config.json`.
StatusOr<TokenizerConfig> ParseTokenizerConfig(std::string_view text);

/// \brief Resolves the checkpoint's special tokens.
///
/// Sources are merged in increasing precedence:
///   1. `tokenizer.json`'s `added_tokens` -- authoritative for ids;
///   2. `special_tokens_map.json`;
///   3. `tokenizer_config.json` (`bos_token` ... and `added_tokens_decoder`);
///   4. explicit overrides from the caller's options.
///
/// A token whose declared id disagrees with the backend's is resolved in
/// the backend's favour, because the backend is what actually produces ids.
/// `warnings` collects those disagreements and anything else a caller running
/// with `strict` should be told about.
StatusOr<SpecialTokens> ResolveSpecialTokens(const LocalArtifacts& artifacts,
                                             const TokenizerConfig& config,
                                             const PretrainedTokenizerOptions& options,
                                             TokenizerBackend& backend,
                                             std::vector<std::string>* warnings);

/// \brief Cross-checks the resolved state against `config.json`.
///
/// Never fatal on its own; findings land in `warnings`, which the caller
/// promotes to an error under `strict`.
void CrossCheckCheckpoint(const LocalArtifacts& artifacts, const SpecialTokens& special,
                          TokenizerBackend& backend, std::vector<std::string>* warnings);

/// \brief Removes the spaces Hugging Face's `clean_up_tokenization_spaces`
///        removes.
std::string CleanUpTokenizationSpaces(std::string_view text);

}  // namespace tokenizer
