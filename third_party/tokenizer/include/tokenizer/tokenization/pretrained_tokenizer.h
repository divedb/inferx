// Adapted from divedb/tokenizer (MIT) for InferX local-only use (ADR 0024).
//
// Removed relative to upstream: Hub/directory/network loading (FromPretrained),
// the SentencePiece backend, batch encode/decode, runtime AddTokens (InferX
// rejects runtime vocabulary mutation), and every hub/cache option. Added:
// byte-driven loading through LocalArtifacts and the upstream streaming
// decode state machine.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tokenizer/chat/chat_message.h"
#include "tokenizer/chat/chat_template.h"
#include "tokenizer/config/pretrained_options.h"
#include "tokenizer/config/tokenizer_config.h"
#include "tokenizer/core/status.h"
#include "tokenizer/core/types.h"
#include "tokenizer/internal/backend/backend.h"
#include "tokenizer/internal/detail/local_artifacts.h"
#include "tokenizer/tokenization/special_tokens.h"

namespace tokenizer {

/// \brief Per-call encoding behaviour: special tokens and truncation.
struct EncodeOptions {
  /// Whether or not to add special tokens when encoding the sequences.
  bool add_special_tokens = true;

  /// Truncate to at most this many ids. `nullopt` means no truncation, even
  /// when the checkpoint declares a `model_max_length`.
  std::optional<int32_t> max_length;

  TruncationSide truncation_side = TruncationSide::kFromConfig;
};

/// \brief Per-call decoding behaviour: special tokens and whitespace cleanup.
struct DecodeOptions {
  bool skip_special_tokens = false;

  /// `nullopt` takes the checkpoint's `clean_up_tokenization_spaces`.
  std::optional<bool> clean_up_tokenization_spaces;
};

/// \brief A tokenizer loaded from checkpoint bytes, modelled on Hugging
///        Face's `PreTrainedTokenizerFast`.
///
/// This is the instance, not a dispatcher: per-family behaviour Hugging Face
/// reaches through `AutoTokenizer` lives in the adapter registry here, so
/// one concrete type serves every checkpoint.
///
/// Tokenization itself belongs to the Rust `tokenizers` engine behind the
/// owned C ABI: the normalizer, pre-tokenizer, model, post-processor,
/// decoder and added-token handling described by `tokenizer.json` are all
/// its work and none of it is reimplemented here. What this class owns is
/// everything `tokenizer.json` does *not* describe -- `tokenizer_config.json`,
/// special-token resolution, chat templates, and the handful of behaviours
/// Hugging Face implements in Python.
///
/// ## Ownership and threading
///
/// **A `PretrainedTokenizer` is not thread-safe. One instance has exactly
/// one owner and must not be called concurrently, including from two
/// `const` calls.** To tokenize in parallel, construct one instance per
/// thread from the same artifact bytes.
///
/// This is the backend's actual contract, not a conservative default: the
/// engine's C ABI is single-handle, and `decode`/`id_to_token` park results
/// in per-call owned buffers but the engine itself is `&mut self` on the
/// Rust side. Exclusive ownership states the true thing.
class PretrainedTokenizer {
 public:
  /// \brief Loads from checkpoint bytes the host already read through its
  ///        own rooted artifact session.
  ///
  /// `artifacts.tokenizer_json` is required; every other member is
  /// optional and its presence merely enables the corresponding behaviour
  /// (special-token roles, chat templates, config cross-checks). This
  /// package performs no filesystem or network access anywhere.
  static StatusOr<std::unique_ptr<PretrainedTokenizer>> FromLocalArtifacts(
      LocalArtifacts artifacts, const PretrainedTokenizerOptions& options = {});

  /// \brief Builds from a serialized `tokenizer.json` held in memory.
  ///
  /// An engine-only loading path: added tokens serialized in
  /// `tokenizer.json` remain available, but their checkpoint-level roles
  /// cannot be recovered without the config artifacts. Use
  /// `FromLocalArtifacts` for the full resolution.
  static StatusOr<std::unique_ptr<PretrainedTokenizer>> FromTokenizerJson(
      std::string tokenizer_json, TokenizerConfig config = {});

  ~PretrainedTokenizer();
  PretrainedTokenizer(const PretrainedTokenizer&) = delete;
  PretrainedTokenizer& operator=(const PretrainedTokenizer&) = delete;
  PretrainedTokenizer(PretrainedTokenizer&&) noexcept;
  PretrainedTokenizer& operator=(PretrainedTokenizer&&) noexcept;

  /// \brief Tokenizes one string into ids, as Hugging Face's
  ///        `tokenizer(text)["input_ids"]` does.
  ///
  /// Truncation happens only when `options.max_length` is set. It is a plain
  /// slice applied *after* the special tokens are in place, so truncating on
  /// the right can drop a trailing EOS and on the left a leading BOS. The
  /// checkpoint's `model_max_length` never truncates on its own; ask for it
  /// explicitly via `ModelMaxLength()` if that is the limit you want.
  ///
  /// \param text    UTF-8 text. An empty string is valid input.
  /// \param options Special-token and truncation behaviour.
  /// \return The token ids, or InvalidArgument when `text` is not valid
  ///         UTF-8 -- the engine cannot accept it and will not guess.
  StatusOr<std::vector<TokenId>> Encode(std::string_view text, const EncodeOptions& options = {});

  /// \brief Turns ids back into text, as Hugging Face's `decode` does.
  ///
  /// Special tokens are kept by default, matching Hugging Face. Enabling
  /// `skip_special_tokens` drops the checkpoint's own special tokens.
  /// Out-of-range ids are InvalidArgument.
  StatusOr<std::string> Decode(absl::Span<const TokenId> ids, const DecodeOptions& options = {});

  /// \brief Creates streaming-decode state driven by the upstream
  ///        `step_decode_stream` algorithm.
  ///
  /// `Push` semantics live on the returned stream: a step returns no chunk
  /// while a split UTF-8/byte-fallback sequence is incomplete, and chunks
  /// already returned never change. The concatenation of all chunks plus
  /// `Finish` equals one-shot `Decode` of the same ids with the same
  /// `skip_special_tokens` and cleanup disabled.
  StatusOr<std::unique_ptr<DecodeStream>> NewDecodeStream(bool skip_special_tokens);

  /// Steps a decode stream on this instance's engine. The stream state is
  /// externalized, so any instance constructed from the same checkpoint
  /// bytes may step the same stream.
  StatusOr<std::optional<std::string>> StepDecodeStream(DecodeStream& stream, TokenId id);

  /// Flushes a decode stream on this instance's engine.
  StatusOr<std::string> FinishDecodeStream(DecodeStream& stream);

  /// \returns `kInvalidTokenId` when the token is not in the vocabulary.
  TokenId TokenToId(std::string_view token);

  /// \returns An empty string when the id is not in the vocabulary.
  std::string IdToToken(TokenId id);

  /// \brief The full id -> token list, sorted by id, in one call.
  StatusOr<std::vector<std::pair<TokenId, std::string>>> VocabDump();

  /// \brief Ids the engine's added vocabulary marks special.
  StatusOr<std::vector<TokenId>> SpecialIds();

  size_t VocabSize();

  bool HasChatTemplate(std::string_view name = "default") const;

  /// \brief Names of the templates this checkpoint ships.
  std::vector<std::string> ChatTemplateNames() const;

  /// \brief What the named template supports. Compiles it if needed.
  StatusOr<const ChatTemplateCaps*> ChatTemplateCapabilities(std::string_view name = "default");

  /// \brief Renders a conversation into prompt text.
  ///
  /// \returns FailedPrecondition when the checkpoint ships no template. There
  ///          is deliberately no built-in default: rendering a conversation in
  ///          the wrong format produces confidently wrong model output, which
  ///          is worse than an error.
  StatusOr<std::string> ApplyChatTemplate(absl::Span<const ChatMessage> messages,
                                          const ChatTemplateOptions& options = {});

  /// \brief `ApplyChatTemplate` followed by encoding, matching Hugging Face's
  /// `apply_chat_template(..., tokenize=True)`.
  ///
  /// Not the same as encoding the result of `ApplyChatTemplate` yourself: the
  /// template already emits the checkpoint's special tokens, so this encodes
  /// with `add_special_tokens=false`. Encoding it the other way double-adds
  /// BOS on Llama-family checkpoints.
  StatusOr<std::vector<TokenId>> ApplyChatTemplateAndEncode(
      absl::Span<const ChatMessage> messages, const ChatTemplateOptions& options = {});

  const TokenizerConfig& GetConfig() const;
  const SpecialTokens& GetSpecialTokens() const;

  std::optional<TokenId> TryGetBosTokenId() const;
  std::optional<TokenId> TryGetEosTokenId() const;
  std::optional<TokenId> TryGetPadTokenId() const;
  std::optional<TokenId> TryGetUnkTokenId() const;

  int64_t ModelMaxLength() const;

  bool IsSpecial(TokenId id) const;

  /// \brief Load-time cross-check warnings collected during
  ///        `FromLocalArtifacts`, for callers that did not request `strict`.
  const std::vector<std::string>& load_warnings() const;

 private:
  class Impl;

  explicit PretrainedTokenizer(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace tokenizer
