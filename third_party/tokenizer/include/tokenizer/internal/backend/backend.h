// Adapted from divedb/tokenizer (MIT) for InferX local-only use (ADR 0024).

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tokenizer/core/status.h"
#include "tokenizer/core/types.h"
#include "tokenizer/tokenization/special_tokens.h"

namespace tokenizer {

class DecodeStream;

/// \brief The seam over the owned InferX tokenizer engine shim.
///
/// Deliberately narrow: it exposes exactly the operations the engine
/// performs. Everything above this line -- config, special tokens, chat
/// templates -- is ours; everything below it is the Hugging Face Rust
/// `tokenizers` engine behind an error-returning C ABI.
///
/// The seam also carries the upstream streaming-decode state machine: a
/// DecodeStream owns the externalized `step_decode_stream` state and borrows
/// the backend for one step at a time, so any pool instance can serve any
/// decoder.
///
/// Implementations are not thread-safe; one instance has exactly one owner.
class TokenizerBackend {
 public:
  virtual ~TokenizerBackend() = default;

  virtual StatusOr<std::vector<TokenId>> Encode(std::string_view text, bool add_special_tokens) = 0;

  virtual StatusOr<std::string> Decode(absl::Span<const TokenId> ids, bool skip_special_tokens) = 0;

  virtual size_t VocabSize() = 0;
  virtual std::string IdToToken(TokenId id) = 0;
  virtual TokenId TokenToId(std::string_view token) = 0;

  /// \brief One-shot dump of the full vocabulary, sorted by id.
  ///
  /// Exists so metadata construction does not cross the FFI once per id.
  virtual StatusOr<std::vector<std::pair<TokenId, std::string>>> VocabDump() = 0;

  /// \brief Ids the engine's added vocabulary marks special.
  virtual StatusOr<std::vector<TokenId>> SpecialIds() = 0;

  /// \brief Creates streaming-decode state bound to this backend's options.
  virtual StatusOr<std::unique_ptr<DecodeStream>> NewDecodeStream(bool skip_special_tokens) = 0;
};

/// \brief Externalized upstream streaming-decode state.
///
/// Holds exactly the state of the upstream `step_decode_stream` free
/// function. `Step` borrows `backend` for one call; no state is kept between
/// backends, so a pool may serve the same stream from different instances.
class DecodeStream {
 public:
  /// \brief One decode step.
  ///
  /// \return `nullopt` when the id alone commits no chunk yet (split code
  ///         point or byte-fallback sequence); the chunk otherwise.
  StatusOr<std::optional<std::string>> Step(TokenizerBackend& backend, TokenId id);

  /// \brief Flushes bytes the engine is still withholding.
  ///
  /// `DataLoss` when the buffered ids decode to an irrecoverably incomplete
  /// trailing sequence.
  StatusOr<std::string> Finish(TokenizerBackend& backend);

  DecodeStream(const DecodeStream&) = delete;
  DecodeStream& operator=(const DecodeStream&) = delete;
  ~DecodeStream();

  // Internal: constructed by the backend implementation from C-ABI state.
  DecodeStream(void* state, bool skip_special_tokens);

 private:
  void* state_ = nullptr;  // IxDecodeStream* from the C ABI.
  bool skip_special_tokens_ = false;
};

/// \brief Creates the engine-backed tokenizer from a `tokenizer.json` blob.
///
/// The owned ABI is error-returning: malformed blobs come back as
/// `DataLoss`/`InvalidArgument` statuses, never as a process abort.
StatusOr<std::unique_ptr<TokenizerBackend>> CreateEngineBackend(const std::string& tokenizer_json);

/// \brief Configures engine-internal parallelism before any handle is used.
///
/// InferX serves single strings from exclusive worker handles, so the
/// engine's internal Rayon pool must stay off the serving path.
void SetEngineParallelism(bool enabled);

}  // namespace tokenizer
