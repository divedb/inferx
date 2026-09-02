// Adapted from divedb/tokenizer (MIT) for InferX local-only use (ADR 0024):
// artifact reads go through LocalArtifacts bytes instead of ModelDir paths.

#include "tokenizer/tokenization/special_tokens.h"

#include <array>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tokenizer/internal/detail/json_bridge.h"
#include "tokenizer/internal/detail/loaders.h"

namespace tokenizer {
namespace {

constexpr std::array<std::string_view, 7> kRoles = {
    "bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"};

std::optional<AddedToken>* SlotFor(SpecialTokens& tokens, std::string_view role) {
  if (role == "bos_token") return &tokens.bos;
  if (role == "eos_token") return &tokens.eos;
  if (role == "unk_token") return &tokens.unk;
  if (role == "sep_token") return &tokens.sep;
  if (role == "pad_token") return &tokens.pad;
  if (role == "cls_token") return &tokens.cls;
  if (role == "mask_token") return &tokens.mask;
  return nullptr;
}

/// Reads one special token, which appears in three shapes across real
/// checkpoints: a bare string (Qwen, gpt-oss), an AddedToken dict with a
/// `__type` marker (DeepSeek-V2-Lite), or absent entirely (gpt2).
std::optional<AddedToken> ParseTokenField(const Json& value) {
  if (value.is_string()) {
    AddedToken token;
    token.content = value.get<std::string>();
    return token;
  }
  if (value.is_object()) {
    auto content = GetString(value, "content");
    if (!content.has_value()) return std::nullopt;
    AddedToken token;
    token.content = *content;
    token.lstrip = GetBool(value, "lstrip").value_or(false);
    token.rstrip = GetBool(value, "rstrip").value_or(false);
    token.normalized = GetBool(value, "normalized").value_or(true);
    token.single_word = GetBool(value, "single_word").value_or(false);
    token.special = GetBool(value, "special").value_or(true);
    if (auto id = GetInt(value, "id")) {
      token.id = static_cast<TokenId>(*id);
    }
    return token;
  }
  return std::nullopt;
}

void MergeRoles(const Json& source, SpecialTokens& tokens) {
  for (std::string_view role : kRoles) {
    auto it = source.find(std::string(role));
    if (it == source.end() || it->is_null()) continue;
    if (auto parsed = ParseTokenField(*it)) {
      *SlotFor(tokens, role) = std::move(*parsed);
    }
  }

  auto additional = source.find("additional_special_tokens");
  if (additional != source.end() && additional->is_array()) {
    tokens.additional.clear();
    for (const Json& entry : *additional) {
      if (auto parsed = ParseTokenField(entry)) {
        tokens.additional.push_back(std::move(*parsed));
      }
    }
  }
}

/// `tokenizer.json`'s `added_tokens` array. Ids here come from the serialized
/// tokenizer itself, so they are the most trustworthy source in the file.
void MergeAddedTokensArray(const Json& tokenizer_json, std::vector<AddedToken>* out) {
  auto added = tokenizer_json.find("added_tokens");
  if (added == tokenizer_json.end() || !added->is_array()) return;
  for (const Json& entry : *added) {
    if (auto parsed = ParseTokenField(entry)) {
      out->push_back(std::move(*parsed));
    }
  }
}

/// `added_tokens_decoder` maps id -> AddedToken, keyed by the id as a string.
void MergeAddedTokensDecoder(const Json& config, std::vector<AddedToken>* out) {
  auto decoder = config.find("added_tokens_decoder");
  if (decoder == config.end() || !decoder->is_object()) return;
  for (auto it = decoder->begin(); it != decoder->end(); ++it) {
    auto parsed = ParseTokenField(it.value());
    if (!parsed.has_value()) continue;
    TokenId id = kInvalidTokenId;
    if (absl::SimpleAtoi(it.key(), &id)) parsed->id = id;
    out->push_back(std::move(*parsed));
  }
}

void RegisterToken(const AddedToken& token, TokenizerBackend& backend, SpecialTokens& tokens,
                   std::vector<std::string>* warnings) {
  if (token.content.empty()) return;

  const TokenId resolved = backend.TokenToId(token.content);
  if (resolved == kInvalidTokenId) {
    if (warnings != nullptr) {
      warnings->push_back(
          absl::StrCat("special token \"", token.content,
                       "\" is declared by the checkpoint but is not in the vocabulary"));
    }
    return;
  }
  if (token.id.has_value() && *token.id != resolved && warnings != nullptr) {
    warnings->push_back(absl::StrCat("special token \"", token.content, "\" is declared with id ",
                                     *token.id, " but the tokenizer resolves it to ", resolved,
                                     "; using ", resolved));
  }
  if (token.special) {
    tokens.token_to_id[token.content] = resolved;
    tokens.all_special_ids.insert(resolved);
  }
}

/// Lowest-precedence source: the model's own config.json.
///
/// Some checkpoints declare their special tokens nowhere else. gpt2 is the
/// example -- its tokenizer_config.json contains only model_max_length, and
/// the reference implementation fills bos/eos/unk from hardcoded Python class
/// defaults. Reproducing *that* would need a per-model table, which is
/// precisely what this design refuses. But config.json's `eos_token_id` is the
/// checkpoint's own declaration, so honouring it is configuration-driven and
/// gets the same answer.
void FillFromModelConfig(const LocalArtifacts& artifacts, TokenizerBackend& backend,
                         SpecialTokens& tokens) {
  if (!artifacts.config.has_value()) return;
  auto parsed = ParseJson(*artifacts.config, "config.json");
  if (!parsed.ok() || !parsed->is_object()) return;

  const std::pair<std::string_view, std::optional<AddedToken>*> roles[] = {
      {"bos_token_id", &tokens.bos},
      {"eos_token_id", &tokens.eos},
      {"pad_token_id", &tokens.pad},
  };
  for (const auto& [key, slot] : roles) {
    if (slot->has_value()) continue;
    auto declared = GetInt(*parsed, key);
    if (!declared.has_value()) continue;
    const auto id = static_cast<TokenId>(*declared);
    std::string content = backend.IdToToken(id);
    if (content.empty()) continue;
    AddedToken token;
    token.content = std::move(content);
    token.id = id;
    *slot = std::move(token);
  }
}

void ResolveRole(std::optional<AddedToken>& slot, TokenizerBackend& backend, SpecialTokens& tokens,
                 std::vector<std::string>* warnings) {
  if (!slot.has_value()) return;
  const TokenId resolved = backend.TokenToId(slot->content);
  if (resolved == kInvalidTokenId) {
    if (warnings != nullptr) {
      warnings->push_back(
          absl::StrCat("special token \"", slot->content,
                       "\" is declared by the checkpoint but is not in the vocabulary"));
    }
    slot->id = std::nullopt;
    return;
  }
  slot->id = resolved;
  tokens.token_to_id[slot->content] = resolved;
  tokens.all_special_ids.insert(resolved);
}

}  // namespace

std::optional<TokenId> SpecialTokens::bos_id() const {
  return bos.has_value() ? bos->id : std::nullopt;
}
std::optional<TokenId> SpecialTokens::eos_id() const {
  return eos.has_value() ? eos->id : std::nullopt;
}
std::optional<TokenId> SpecialTokens::pad_id() const {
  return pad.has_value() ? pad->id : std::nullopt;
}
std::optional<TokenId> SpecialTokens::unk_id() const {
  return unk.has_value() ? unk->id : std::nullopt;
}

StatusOr<SpecialTokens> ResolveSpecialTokens(const LocalArtifacts& artifacts,
                                             const TokenizerConfig& config,
                                             const PretrainedTokenizerOptions& options,
                                             TokenizerBackend& backend,
                                             std::vector<std::string>* warnings) {
  SpecialTokens tokens;
  std::vector<AddedToken> catalogue;

  // 1. tokenizer.json's added_tokens -- the ids the engine actually uses.
  {
    auto parsed = ParseJson(artifacts.tokenizer_json, "tokenizer.json");
    if (parsed.ok()) MergeAddedTokensArray(*parsed, &catalogue);
  }

  // 2. special_tokens_map.json.
  if (artifacts.special_tokens_map.has_value()) {
    auto parsed = ParseJson(*artifacts.special_tokens_map, "special_tokens_map.json");
    if (parsed.ok() && parsed->is_object()) MergeRoles(*parsed, tokens);
  }

  // 3. tokenizer_config.json -- both the role fields and the id-keyed
  //    added_tokens_decoder map.
  const Json& raw = JsonBridge::ToInternal(config.raw);
  if (raw.is_object()) {
    MergeRoles(raw, tokens);
    MergeAddedTokensDecoder(raw, &catalogue);
  }

  // 4. Explicit overrides win over anything the checkpoint declares.
  for (const auto& [role, content] : options.special_token_overrides) {
    std::optional<AddedToken>* slot = SlotFor(tokens, role);
    if (slot == nullptr) {
      return InvalidArgumentError("unknown special token role \"", role, "\"");
    }
    AddedToken token;
    token.content = content;
    *slot = std::move(token);
  }

  // 5. Anything still unset falls back to the model's own config.json.
  FillFromModelConfig(artifacts, backend, tokens);

  for (const AddedToken& token : catalogue) {
    RegisterToken(token, backend, tokens, warnings);
  }
  ResolveRole(tokens.bos, backend, tokens, warnings);
  ResolveRole(tokens.eos, backend, tokens, warnings);
  ResolveRole(tokens.unk, backend, tokens, warnings);
  ResolveRole(tokens.sep, backend, tokens, warnings);
  ResolveRole(tokens.pad, backend, tokens, warnings);
  ResolveRole(tokens.cls, backend, tokens, warnings);
  ResolveRole(tokens.mask, backend, tokens, warnings);
  for (AddedToken& token : tokens.additional) {
    std::optional<AddedToken> slot = token;
    ResolveRole(slot, backend, tokens, warnings);
    token = *slot;
  }

  return tokens;
}

void CrossCheckCheckpoint(const LocalArtifacts& artifacts, const SpecialTokens& special,
                          TokenizerBackend& backend, std::vector<std::string>* warnings) {
  if (warnings == nullptr || !artifacts.config.has_value()) return;

  auto parsed = ParseJson(*artifacts.config, "config.json");
  if (!parsed.ok() || !parsed->is_object()) return;

  if (auto declared = GetInt(*parsed, "eos_token_id")) {
    const std::optional<TokenId> resolved = special.eos_id();
    if (resolved.has_value() && *declared != *resolved) {
      warnings->push_back(absl::StrCat("config.json declares eos_token_id ", *declared,
                                       " but the tokenizer resolves the eos token to ", *resolved,
                                       "; generation_config.json may list several stop ids"));
    }
  }
  if (auto vocab = GetInt(*parsed, "vocab_size")) {
    const auto backend_size = static_cast<int64_t>(backend.VocabSize());
    if (*vocab > backend_size) {
      warnings->push_back(absl::StrCat("config.json declares vocab_size ", *vocab,
                                       " but the tokenizer only has ", backend_size, " tokens"));
    }
  }
}

}  // namespace tokenizer
