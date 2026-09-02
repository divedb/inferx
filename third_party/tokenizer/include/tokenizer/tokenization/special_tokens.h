#pragma once

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tokenizer/core/types.h"

namespace tokenizer {

/// \brief One entry of `tokenizer.json`'s `added_tokens`, or the dict form a
///        special token takes in `tokenizer_config.json`.
struct AddedToken {
  std::string content;
  bool lstrip = false;
  bool rstrip = false;
  bool normalized = true;
  bool single_word = false;
  bool special = true;

  /// Present when the source declared an id. The backend's own mapping wins
  /// on disagreement, because the backend is what actually produces ids.
  std::optional<TokenId> id;
};

/// \brief The special tokens a checkpoint declares, resolved to ids.
///
/// Assembled from three sources with documented precedence -- see
/// `SpecialTokensBuilder` in the implementation. Callers only read it.
struct SpecialTokens {
  std::optional<AddedToken> bos;
  std::optional<AddedToken> eos;
  std::optional<AddedToken> unk;
  std::optional<AddedToken> sep;
  std::optional<AddedToken> pad;
  std::optional<AddedToken> cls;
  std::optional<AddedToken> mask;

  /// `additional_special_tokens`, in declaration order.
  std::vector<AddedToken> additional;

  /// Every token the checkpoint marks special, including `added_tokens`
  /// entries that are not one of the seven named roles above.
  absl::flat_hash_map<std::string, TokenId> token_to_id;
  absl::flat_hash_set<TokenId> all_special_ids;

  std::optional<TokenId> bos_id() const;
  std::optional<TokenId> eos_id() const;
  std::optional<TokenId> pad_id() const;
  std::optional<TokenId> unk_id() const;

  bool IsSpecial(TokenId id) const { return all_special_ids.contains(id); }
};

}  // namespace tokenizer
