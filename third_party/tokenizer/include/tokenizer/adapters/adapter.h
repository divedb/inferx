#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tokenizer/config/tokenizer_config.h"
#include "tokenizer/core/status.h"
#include "tokenizer/core/types.h"
#include "tokenizer/tokenization/special_tokens.h"

namespace tokenizer {

struct EncodeOptions;
struct DecodeOptions;

/// \brief A hook for behaviour Hugging Face implements in a Python subclass
///        and which therefore has no equivalent in the Rust pipeline.
///
/// Adapters are selected by `tokenizer_config.json`'s `tokenizer_class`, never
/// by a model name or a path substring: that is the same signal Hugging Face
/// itself dispatches on, it is present in every real checkpoint, and it
/// survives a directory being renamed.
///
/// Keep these small. An adapter is the escape hatch for a documented upstream
/// behaviour, not a place to put model-specific tokenization.
class TokenizerAdapter {
 public:
  virtual ~TokenizerAdapter() = default;

  virtual std::string_view name() const = 0;

  /// \brief Last chance to adjust resolved state, at the end of loading.
  virtual Status Finalize(TokenizerConfig& config, SpecialTokens& special_tokens) const {
    (void)config;
    (void)special_tokens;
    return OkStatus();
  }

  /// \brief Applied to an encoding after the backend encodes.
  virtual void PostEncode(std::vector<TokenId>& ids, const EncodeOptions& options,
                          const TokenizerConfig& config,
                          const SpecialTokens& special_tokens) const {
    (void)ids;
    (void)options;
    (void)config;
    (void)special_tokens;
  }

  /// \brief Applied to text after the backend decodes.
  virtual void PostDecode(std::string& text, const DecodeOptions& options,
                          const TokenizerConfig& config) const {
    (void)text;
    (void)options;
    (void)config;
  }
};

/// \brief Maps `tokenizer_class` to an adapter.
///
/// Registration is one line in a .cc, so a downstream user can support a new
/// family without touching a header -- and `PretrainedTokenizer` itself
/// contains no per-model branching at all.
class AdapterRegistry {
 public:
  static AdapterRegistry& Instance();

  void Register(std::string tokenizer_class, std::shared_ptr<const TokenizerAdapter> adapter);

  /// \brief Never null: an unknown class gets the default adapter, which
  ///        implements only the config-driven behaviour every checkpoint
  ///        shares.
  std::shared_ptr<const TokenizerAdapter> For(std::string_view tokenizer_class) const;

 private:
  AdapterRegistry();

  absl::flat_hash_map<std::string, std::shared_ptr<const TokenizerAdapter>> adapters_;
  std::shared_ptr<const TokenizerAdapter> default_adapter_;
};

}  // namespace tokenizer
