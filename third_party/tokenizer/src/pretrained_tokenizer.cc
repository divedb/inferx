// Adapted from divedb/tokenizer (MIT) for InferX local-only use (ADR 0024):
// the loading path consumes LocalArtifacts bytes; Hub resolution, directory
// scanning, SentencePiece, batch encode/decode, and runtime AddTokens are
// removed.

#include "tokenizer/tokenization/pretrained_tokenizer.h"

#include <algorithm>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tokenizer/adapters/adapter.h"
#include "tokenizer/internal/adapters/default_adapter.h"
#include "tokenizer/internal/backend/backend.h"
#include "tokenizer/internal/chat/template_source.h"
#include "tokenizer/internal/detail/json_bridge.h"
#include "tokenizer/internal/detail/loaders.h"

namespace tokenizer {
namespace {

Status NoChatTemplateError() {
  Status status = FailedPreconditionError(
      "this checkpoint defines no chat template",
      "; supply one with PretrainedTokenizerOptions::chat_template_override, "
      "or call Encode() directly. There is deliberately no built-in default: "
      "rendering a conversation in the wrong format produces confidently "
      "wrong model output.");
  return status;
}

void Truncate(std::vector<TokenId>& ids, int32_t max_length, TruncationSide side,
              const TokenizerConfig& config) {
  if (max_length < 0 || ids.size() <= static_cast<size_t>(max_length)) return;

  TruncationSide effective = side;

  if (effective == TruncationSide::kFromConfig) {
    effective = config.truncation_side == "left" ? TruncationSide::kLeft : TruncationSide::kRight;
  }

  if (effective == TruncationSide::kLeft) {
    ids.erase(ids.begin(), ids.end() - max_length);
  } else {
    ids.resize(static_cast<size_t>(max_length));
  }
}

/// tokenizer_config.json is optional -- gpt2 ships one containing only
/// model_max_length, and a bare checkpoint ships none at all and must still
/// load.
StatusOr<TokenizerConfig> LoadTokenizerConfig(const LocalArtifacts& artifacts,
                                              const PretrainedTokenizerOptions& options,
                                              std::vector<std::string>* warnings) {
  TokenizerConfig config;
  if (artifacts.tokenizer_config.has_value()) {
    ABSL_ASSIGN_OR_RETURN(config, ParseTokenizerConfig(*artifacts.tokenizer_config));
  } else {
    warnings->push_back(
        "the checkpoint has no tokenizer_config.json; special tokens and the "
        "chat template will both be unavailable");
  }

  if (options.tokenizer_type.has_value()) {
    config.tokenizer_class = *options.tokenizer_type;
  }
  return config;
}

/// Sources only. Compiling a template probes its capabilities by rendering
/// it several times, which a caller who only encodes text should not pay
/// for, so that is deferred to first use.
ChatTemplateSources CollectChatTemplates(const LocalArtifacts& artifacts,
                                         const TokenizerConfig& config,
                                         const PretrainedTokenizerOptions& options) {
  ChatTemplateSources sources =
      DiscoverChatTemplates(artifacts, JsonBridge::ToInternal(config.raw));
  if (options.chat_template_override.has_value()) {
    sources.by_name["default"] = *options.chat_template_override;
    sources.default_name = "default";
  }
  return sources;
}

}  // namespace

class PretrainedTokenizer::Impl {
 public:
  std::unique_ptr<TokenizerBackend> backend;
  TokenizerConfig config;
  SpecialTokens special;
  std::shared_ptr<const TokenizerAdapter> adapter;
  ChatTemplateSources template_sources;
  std::vector<std::string> load_warnings;

  // Compiled lazily: constructing a template probes its capabilities by
  // rendering it several times, which a caller who only encodes text should
  // not pay for.
  absl::flat_hash_map<std::string, ChatTemplate> compiled;

  Status ApplyChatTemplateSource(std::string_view name, const ChatTemplate** out);
};

Status PretrainedTokenizer::Impl::ApplyChatTemplateSource(std::string_view name,
                                                          const ChatTemplate** out) {
  auto compiled_it = compiled.find(name);

  if (compiled_it != compiled.end()) {
    *out = &compiled_it->second;

    return OkStatus();
  }

  const std::string* source = template_sources.Find(name);

  if (source == nullptr) {
    if (template_sources.empty()) return NoChatTemplateError();

    return NotFoundError("no chat template named \"", name, "\"; this checkpoint provides ",
                         absl::StrJoin(template_sources.Names(), ", "));
  }

  const std::string bos = special.bos.has_value() ? special.bos->content : std::string();
  const std::string eos = special.eos.has_value() ? special.eos->content : std::string();

  ABSL_ASSIGN_OR_RETURN(ChatTemplate tmpl, ChatTemplate::Create(*source, bos, eos));
  auto [it, inserted] = compiled.emplace(std::string(name), std::move(tmpl));
  *out = &it->second;

  return OkStatus();
}

PretrainedTokenizer::PretrainedTokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PretrainedTokenizer::~PretrainedTokenizer() = default;
PretrainedTokenizer::PretrainedTokenizer(PretrainedTokenizer&&) noexcept = default;
PretrainedTokenizer& PretrainedTokenizer::operator=(PretrainedTokenizer&&) noexcept = default;

StatusOr<std::unique_ptr<PretrainedTokenizer>> PretrainedTokenizer::FromLocalArtifacts(
    LocalArtifacts artifacts, const PretrainedTokenizerOptions& options) {
  auto impl = std::make_unique<Impl>();

  // Step 1: the tokenization engine, from bytes only.
  ABSL_ASSIGN_OR_RETURN(impl->backend, CreateEngineBackend(artifacts.tokenizer_json));

  // Step 2: tokenizer_config.json, plus the caller's tokenizer_type override.
  ABSL_ASSIGN_OR_RETURN(impl->config,
                        LoadTokenizerConfig(artifacts, options, &impl->load_warnings));

  // Step 3: special tokens, merged across their sources and cross-checked
  // against config.json. Disagreements become warnings, not failures.
  ABSL_ASSIGN_OR_RETURN(impl->special, ResolveSpecialTokens(artifacts, impl->config, options,
                                                            *impl->backend, &impl->load_warnings));
  CrossCheckCheckpoint(artifacts, impl->special, *impl->backend, &impl->load_warnings);

  // Step 4: the family adapter, selected by tokenizer_class.
  impl->adapter = AdapterRegistry::Instance().For(impl->config.tokenizer_class);

  // Step 5: chat template sources. Compilation is deferred to first use.
  impl->template_sources = CollectChatTemplates(artifacts, impl->config, options);

  // Step 6: the adapter's last chance to adjust what loading resolved.
  ABSL_RETURN_IF_ERROR(impl->adapter->Finalize(impl->config, impl->special));

  // Step 7: under strict, everything the load glossed over becomes the error.
  if (options.strict && !impl->load_warnings.empty()) {
    return absl::DataLossError(
        absl::StrCat("the checkpoint did not load cleanly and strict was requested: ",
                     absl::StrJoin(impl->load_warnings, "; ")));
  }

  return std::unique_ptr<PretrainedTokenizer>(new PretrainedTokenizer(std::move(impl)));
}

StatusOr<std::unique_ptr<PretrainedTokenizer>> PretrainedTokenizer::FromTokenizerJson(
    std::string tokenizer_json, TokenizerConfig config) {
  auto impl = std::make_unique<Impl>();
  ABSL_ASSIGN_OR_RETURN(impl->backend, CreateEngineBackend(tokenizer_json));
  impl->config = std::move(config);
  impl->adapter = AdapterRegistry::Instance().For(impl->config.tokenizer_class);

  // tokenizer.json carries added_tokens, so the special-token set can still
  // be recovered without a checkpoint directory. It does not assign those
  // tokens checkpoint-level roles such as BOS or PAD.
  auto parsed = ParseJson(tokenizer_json, "tokenizer.json");
  if (parsed.ok() && parsed->is_object()) {
    auto added = parsed->find("added_tokens");
    if (added != parsed->end() && added->is_array()) {
      for (const Json& entry : *added) {
        auto content = GetString(entry, "content");
        if (!content.has_value()) continue;
        if (!GetBool(entry, "special").value_or(true)) continue;
        const TokenId id = impl->backend->TokenToId(*content);
        if (id == kInvalidTokenId) continue;
        impl->special.token_to_id[*content] = id;
        impl->special.all_special_ids.insert(id);
      }
    }
  }
  return std::unique_ptr<PretrainedTokenizer>(new PretrainedTokenizer(std::move(impl)));
}

StatusOr<std::vector<TokenId>> PretrainedTokenizer::Encode(std::string_view text,
                                                           const EncodeOptions& options) {
  ABSL_ASSIGN_OR_RETURN(std::vector<TokenId> ids,
                        impl_->backend->Encode(text, options.add_special_tokens));
  impl_->adapter->PostEncode(ids, options, impl_->config, impl_->special);

  if (options.max_length.has_value()) {
    Truncate(ids, *options.max_length, options.truncation_side, impl_->config);
  }

  return ids;
}

StatusOr<std::string> PretrainedTokenizer::Decode(absl::Span<const TokenId> ids,
                                                  const DecodeOptions& options) {
  for (TokenId id : ids) {
    if (id < 0) {
      return InvalidArgumentError("token id ", id, " is negative and cannot be decoded");
    }
  }
  ABSL_ASSIGN_OR_RETURN(std::string text, impl_->backend->Decode(ids, options.skip_special_tokens));
  impl_->adapter->PostDecode(text, options, impl_->config);
  return text;
}

StatusOr<std::unique_ptr<DecodeStream>> PretrainedTokenizer::NewDecodeStream(
    bool skip_special_tokens) {
  return impl_->backend->NewDecodeStream(skip_special_tokens);
}

StatusOr<std::optional<std::string>> PretrainedTokenizer::StepDecodeStream(DecodeStream& stream,
                                                                           TokenId id) {
  return stream.Step(*impl_->backend, id);
}

StatusOr<std::string> PretrainedTokenizer::FinishDecodeStream(DecodeStream& stream) {
  return stream.Finish(*impl_->backend);
}

TokenId PretrainedTokenizer::TokenToId(std::string_view token) {
  return impl_->backend->TokenToId(token);
}

std::string PretrainedTokenizer::IdToToken(TokenId id) { return impl_->backend->IdToToken(id); }

StatusOr<std::vector<std::pair<TokenId, std::string>>> PretrainedTokenizer::VocabDump() {
  return impl_->backend->VocabDump();
}

StatusOr<std::vector<TokenId>> PretrainedTokenizer::SpecialIds() {
  return impl_->backend->SpecialIds();
}

size_t PretrainedTokenizer::VocabSize() { return impl_->backend->VocabSize(); }

bool PretrainedTokenizer::HasChatTemplate(std::string_view name) const {
  return impl_->template_sources.Find(name) != nullptr;
}

std::vector<std::string> PretrainedTokenizer::ChatTemplateNames() const {
  return impl_->template_sources.Names();
}

StatusOr<const ChatTemplateCaps*> PretrainedTokenizer::ChatTemplateCapabilities(
    std::string_view name) {
  const ChatTemplate* tmpl = nullptr;
  ABSL_RETURN_IF_ERROR(impl_->ApplyChatTemplateSource(name, &tmpl));

  return &tmpl->caps();
}

StatusOr<std::string> PretrainedTokenizer::ApplyChatTemplate(absl::Span<const ChatMessage> messages,
                                                             const ChatTemplateOptions& options) {
  const ChatTemplate* tmpl = nullptr;
  ABSL_RETURN_IF_ERROR(impl_->ApplyChatTemplateSource(options.template_name, &tmpl));
  return tmpl->Apply(messages, options);
}

StatusOr<std::vector<TokenId>> PretrainedTokenizer::ApplyChatTemplateAndEncode(
    absl::Span<const ChatMessage> messages, const ChatTemplateOptions& options) {
  ABSL_ASSIGN_OR_RETURN(std::string prompt, ApplyChatTemplate(messages, options));
  // The template has already emitted the checkpoint's special tokens, so
  // encoding must not add them again. Doing otherwise double-adds BOS on
  // Llama-family checkpoints -- this is what Hugging Face's
  // apply_chat_template(..., tokenize=True) does internally.
  EncodeOptions encode;
  encode.add_special_tokens = false;

  return Encode(prompt, encode);
}

const TokenizerConfig& PretrainedTokenizer::GetConfig() const { return impl_->config; }

const SpecialTokens& PretrainedTokenizer::GetSpecialTokens() const { return impl_->special; }

std::optional<TokenId> PretrainedTokenizer::TryGetBosTokenId() const {
  return impl_->special.bos_id();
}

std::optional<TokenId> PretrainedTokenizer::TryGetEosTokenId() const {
  return impl_->special.eos_id();
}

std::optional<TokenId> PretrainedTokenizer::TryGetPadTokenId() const {
  return impl_->special.pad_id();
}

std::optional<TokenId> PretrainedTokenizer::TryGetUnkTokenId() const {
  return impl_->special.unk_id();
}

int64_t PretrainedTokenizer::ModelMaxLength() const { return impl_->config.model_max_length; }

bool PretrainedTokenizer::IsSpecial(TokenId id) const { return impl_->special.IsSpecial(id); }

const std::vector<std::string>& PretrainedTokenizer::load_warnings() const {
  return impl_->load_warnings;
}

}  // namespace tokenizer
