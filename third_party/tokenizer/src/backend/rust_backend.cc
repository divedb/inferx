// Binding of the owned InferX tokenizer C ABI (ADR 0024).
//
// Adapted from divedb/tokenizer's rust_backend.cc (MIT): the ABI it bound
// aborted the process on malformed `tokenizer.json` (`Tokenizer::from_str`
// behind `unwrap()`), aliased decode results to handle-owned scratch, and
// had no streaming decode state. The ABI in inferx_abi.h is owned by this
// repository and fixes all three, so this class is a plain translation of
// statuses and owned buffers.

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx_abi.h"
#include "tokenizer/internal/backend/backend.h"
#include "tokenizer/internal/detail/json_bridge.h"

namespace tokenizer {
namespace {

/// Owns one C-ABI error record for the duration of a call and frees its
/// message on scope exit.
class ScopedError {
 public:
  ScopedError() = default;
  ScopedError(const ScopedError&) = delete;
  ScopedError& operator=(const ScopedError&) = delete;
  ~ScopedError() { ixtok_free_error(&err_); }

  IxError* get() { return &err_; }

  absl::Status Status(const char* what) const {
    std::string message(err_.message ? reinterpret_cast<const char*>(err_.message) : "",
                        err_.message ? static_cast<size_t>(err_.message_len) : 0);
    absl::StatusCode code = absl::StatusCode::kInternal;
    switch (err_.code) {
      case IX_INVALID_ARGUMENT:
        code = absl::StatusCode::kInvalidArgument;
        break;
      case IX_NOT_FOUND:
        code = absl::StatusCode::kNotFound;
        break;
      case IX_DATA_LOSS:
        code = absl::StatusCode::kDataLoss;
        break;
      case IX_PENDING:
      case IX_INTERNAL:
      case IX_PANIC:
      default:
        code = absl::StatusCode::kInternal;
        break;
    }
    if (err_.code == IX_PANIC) {
      // A panic crossing the shim is an engine bug, not a data problem; make
      // that visible instead of a generic internal error.
      return absl::InternalError(absl::StrCat(what, ": the tokenizer engine panicked: ", message));
    }
    return absl::Status(code, absl::StrCat(what, ": ", message));
  }

 private:
  IxError err_{IX_OK, nullptr, 0};
};

/// Owns one C-ABI byte buffer.
class ScopedBytes {
 public:
  ScopedBytes() = default;
  ScopedBytes(const ScopedBytes&) = delete;
  ScopedBytes& operator=(const ScopedBytes&) = delete;
  ~ScopedBytes() { ixtok_free_bytes(ptr_, len_); }

  uint8_t** ptr_slot() { return &ptr_; }
  uintptr_t* len_slot() { return &len_; }

  std::string Take() {
    std::string out(ptr_ ? reinterpret_cast<const char*>(ptr_) : "", static_cast<size_t>(len_));
    ixtok_free_bytes(ptr_, len_);
    ptr_ = nullptr;
    len_ = 0;
    return out;
  }

 private:
  uint8_t* ptr_ = nullptr;
  uintptr_t len_ = 0;
};

/// Owns one C-ABI id buffer.
class ScopedIds {
 public:
  ScopedIds() = default;
  ScopedIds(const ScopedIds&) = delete;
  ScopedIds& operator=(const ScopedIds&) = delete;
  ~ScopedIds() { ixtok_free_ids(ptr_, len_); }

  uint32_t** ptr_slot() { return &ptr_; }
  uintptr_t* len_slot() { return &len_; }

  std::vector<TokenId> Take() {
    std::vector<TokenId> out(ptr_ ? ptr_ : nullptr, ptr_ ? ptr_ + len_ : nullptr);
    ixtok_free_ids(ptr_, len_);
    ptr_ = nullptr;
    len_ = 0;
    return out;
  }

 private:
  uint32_t* ptr_ = nullptr;
  uintptr_t len_ = 0;
};

class RustBackend : public TokenizerBackend {
 public:
  explicit RustBackend(IxTokenizer* handle) : handle_(handle) {}

  RustBackend(const RustBackend&) = delete;
  RustBackend& operator=(const RustBackend&) = delete;

  ~RustBackend() override {
    if (handle_ != nullptr) ixtok_free(handle_);
  }

  StatusOr<std::vector<TokenId>> Encode(std::string_view text, bool add_special_tokens) override {
    ScopedError err;
    ScopedIds ids;
    const int32_t status =
        ixtok_encode(handle_, reinterpret_cast<const uint8_t*>(text.data()), text.size(),
                     add_special_tokens ? 1 : 0, ids.ptr_slot(), ids.len_slot(), err.get());
    if (status != IX_OK && status != IX_PENDING) {
      return err.Status("encode");
    }
    return ids.Take();
  }

  StatusOr<std::string> Decode(absl::Span<const TokenId> ids, bool skip_special_tokens) override {
    if (ids.empty()) return std::string();
    ScopedError err;
    ScopedBytes bytes;
    const int32_t status =
        ixtok_decode(handle_, reinterpret_cast<const uint32_t*>(ids.data()), ids.size(),
                     skip_special_tokens ? 1 : 0, bytes.ptr_slot(), bytes.len_slot(), err.get());
    if (status != IX_OK) return err.Status("decode");
    return bytes.Take();
  }

  size_t VocabSize() override { return ixtok_vocab_size(handle_); }

  std::string IdToToken(TokenId id) override {
    if (id < 0) return std::string();
    ScopedError err;
    ScopedBytes bytes;
    const int32_t status = ixtok_id_to_token(handle_, static_cast<uint32_t>(id), bytes.ptr_slot(),
                                             bytes.len_slot(), err.get());
    if (status == IX_NOT_FOUND) return std::string();
    if (status != IX_OK) return std::string();
    return bytes.Take();
  }

  TokenId TokenToId(std::string_view token) override {
    ScopedError err;
    uint32_t id = 0;
    const int32_t status = ixtok_token_to_id(
        handle_, reinterpret_cast<const uint8_t*>(token.data()), token.size(), &id, err.get());
    if (status == IX_NOT_FOUND) return kInvalidTokenId;
    if (status != IX_OK) return kInvalidTokenId;
    return static_cast<TokenId>(id);
  }

  StatusOr<std::vector<std::pair<TokenId, std::string>>> VocabDump() override {
    ScopedError err;
    ScopedIds ids;
    uint64_t* offsets = nullptr;
    uint8_t* chars = nullptr;
    uintptr_t chars_len = 0;
    uintptr_t count = 0;
    const int32_t status =
        ixtok_vocab_dump(handle_, ids.ptr_slot(), &offsets, &chars, &chars_len, &count, err.get());
    if (status != IX_OK) return err.Status("vocab dump");

    std::vector<std::pair<TokenId, std::string>> vocab;
    vocab.reserve(count);
    for (uintptr_t i = 0; i < count; ++i) {
      const uint64_t begin = offsets[i];
      const uint64_t end = offsets[i + 1];
      vocab.emplace_back(static_cast<TokenId>((*ids.ptr_slot())[i]),
                         std::string(reinterpret_cast<const char*>(chars) + begin,
                                     static_cast<size_t>(end - begin)));
    }
    ixtok_free_offsets(offsets, static_cast<uintptr_t>(count) + 1);
    ixtok_free_bytes(chars, chars_len);
    return vocab;
  }

  StatusOr<std::vector<TokenId>> SpecialIds() override {
    ScopedError err;
    ScopedIds ids;
    const int32_t status = ixtok_special_ids(handle_, ids.ptr_slot(), ids.len_slot(), err.get());
    if (status != IX_OK) return err.Status("special ids");
    return ids.Take();
  }

  StatusOr<std::unique_ptr<DecodeStream>> NewDecodeStream(bool skip_special_tokens) override {
    void* state = ixtok_stream_new(skip_special_tokens ? 1 : 0);
    if (state == nullptr) {
      return InternalError("the tokenizer engine returned no stream state");
    }
    return std::unique_ptr<DecodeStream>(new DecodeStream(state, skip_special_tokens));
  }

 private:
  friend IxTokenizer* AbiHandleOf(TokenizerBackend& backend);

  IxTokenizer* handle_ = nullptr;
};

IxTokenizer* AbiHandleOf(TokenizerBackend& backend) {
  return static_cast<RustBackend&>(backend).handle_;
}

}  // namespace

StatusOr<std::unique_ptr<TokenizerBackend>> CreateEngineBackend(const std::string& tokenizer_json) {
  // Belt-and-braces pre-screen before the FFI: the owned ABI is
  // error-returning, but rejecting obviously-not-a-tokenizer blobs here
  // keeps the error message shape stable for the layers above.
  ABSL_ASSIGN_OR_RETURN(Json parsed, ParseJson(tokenizer_json, "tokenizer.json"));
  if (!parsed.is_object() || !parsed.contains("model")) {
    return absl::DataLossError(
        "tokenizer.json has no \"model\" component; the file is not a "
        "serialized tokenizer");
  }

  ScopedError err;
  IxTokenizer* handle = ixtok_new(reinterpret_cast<const uint8_t*>(tokenizer_json.data()),
                                  tokenizer_json.size(), err.get());
  if (handle == nullptr) {
    return err.Status("tokenizer construction");
  }
  return std::unique_ptr<TokenizerBackend>(new RustBackend(handle));
}

void SetEngineParallelism(bool enabled) { ixtok_set_parallelism(enabled ? 1 : 0); }

DecodeStream::DecodeStream(void* state, bool skip_special_tokens)
    : state_(state), skip_special_tokens_(skip_special_tokens) {}

DecodeStream::~DecodeStream() { ixtok_stream_free(static_cast<IxDecodeStream*>(state_)); }

StatusOr<std::optional<std::string>> DecodeStream::Step(TokenizerBackend& backend, TokenId id) {
  if (id < 0) {
    return InvalidArgumentError("token id is negative");
  }
  ScopedError err;
  ScopedBytes bytes;
  const int32_t status =
      ixtok_stream_step(AbiHandleOf(backend), static_cast<IxDecodeStream*>(state_),
                        static_cast<uint32_t>(id), bytes.ptr_slot(), bytes.len_slot(), err.get());
  if (status == IX_PENDING) return std::nullopt;
  if (status != IX_OK) return err.Status("stream step");
  return std::optional<std::string>(bytes.Take());
}

StatusOr<std::string> DecodeStream::Finish(TokenizerBackend& backend) {
  ScopedError err;
  ScopedBytes bytes;
  const int32_t status =
      ixtok_stream_finish(AbiHandleOf(backend), static_cast<IxDecodeStream*>(state_),
                          bytes.ptr_slot(), bytes.len_slot(), err.get());
  if (status != IX_OK) return err.Status("stream finish");
  return bytes.Take();
}

}  // namespace tokenizer
