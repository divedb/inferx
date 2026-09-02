// The qualified tokenizer adapter.
//
// This is the only translation unit that includes the vendored tokenizer
// package. It maps vendor statuses to InferX statuses, copies engine output
// into owned values, derives TokenizerMetadata from the same engine
// resolution encode uses, and lends exclusive engine instances to callers.
// No vendor type appears in any InferX header.

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/id.h"
#include "inferx/tokenization/incremental_decoder.h"
#include "inferx/tokenization/tokenizer.h"
#include "inferx/tokenization/tokenizer_metadata.h"
#include "tokenizer/config/tokenizer_config.h"
#include "tokenizer/core/status.h"
#include "tokenizer/core/types.h"
#include "tokenizer/internal/detail/json_bridge.h"
#include "tokenizer/internal/detail/local_artifacts.h"
#include "tokenizer/tokenization/pretrained_tokenizer.h"

namespace inferx::tokenization {
namespace {

constexpr char kBackend[] = "inferx-tokenizers-c";
constexpr char kEngineVersion[] = "hf-tokenizers-0.21.2";

// The vendored tokenizers engine is single-handle and thread-affine; every
// vendor handle below is constructed here, lent to exactly one caller at a
// time, and destroyed with the facade.
using VendorTokenizer = ::tokenizer::PretrainedTokenizer;

absl::Status AdaptStatus(const ::tokenizer::Status& status) {
  if (status.ok()) return absl::OkStatus();
  return status;
}

// ---------------------------------------------------------------------------
// Exclusive instance lending
// ---------------------------------------------------------------------------

struct SharedEngine {
  TokenizerMetadata metadata;
  std::string canonical_record;
  std::deque<std::unique_ptr<VendorTokenizer>> idle ABSL_GUARDED_BY(mutex);
  size_t borrowed ABSL_GUARDED_BY(mutex) = 0;
  mutable std::mutex mutex;
  mutable std::condition_variable idle_cv;

  // Waits for an idle exclusive instance. Encode/decode/stream calls are
  // short, so waiting (rather than failing) is the facade's contract: the
  // caller observes logical constness, the engine observes exclusivity.
  std::unique_ptr<VendorTokenizer> Acquire(std::stop_token stop = {}) {
    std::unique_lock<std::mutex> lock(mutex);
    idle_cv.wait(lock,
                 [&] { return !idle.empty() || (stop.stop_possible() && stop.stop_requested()); });
    if (idle.empty()) return nullptr;  // stopped before an instance freed up
    std::unique_ptr<VendorTokenizer> instance = std::move(idle.front());
    idle.pop_front();
    ++borrowed;
    return instance;
  }

  void Release(std::unique_ptr<VendorTokenizer> instance) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      --borrowed;
      idle.push_front(std::move(instance));
    }
    idle_cv.notify_one();
  }
};

// Borrows one exclusive instance for the duration of a single call.
class InstanceLease {
 public:
  InstanceLease(const std::shared_ptr<SharedEngine>& engine, absl::Status& status)
      : engine_(engine), instance_(engine->Acquire()) {
    if (instance_ == nullptr) {
      status = absl::CancelledError("tokenizer acquisition stopped before an instance was free");
    }
  }

  ~InstanceLease() {
    if (instance_ != nullptr) engine_->Release(std::move(instance_));
  }

  VendorTokenizer* operator->() const { return instance_.get(); }
  VendorTokenizer& operator*() const { return *instance_; }
  explicit operator bool() const { return instance_ != nullptr; }

 private:
  std::shared_ptr<SharedEngine> engine_;
  std::unique_ptr<VendorTokenizer> instance_;
};

::tokenizer::EncodeOptions ToVendorEncode(const EncodeOptions& options) {
  ::tokenizer::EncodeOptions vendor;
  vendor.add_special_tokens = options.add_special_tokens;
  if (options.max_tokens.has_value()) {
    vendor.max_length = static_cast<int32_t>(options.max_tokens->value());
  }
  vendor.truncation_side = options.truncation_side == TruncationSide::kLeft
                               ? ::tokenizer::TruncationSide::kLeft
                               : ::tokenizer::TruncationSide::kRight;
  return vendor;
}

// ---------------------------------------------------------------------------
// Metadata derivation
// ---------------------------------------------------------------------------

// Component type names from the serialized pipeline: the exact bytes the
// engine itself parsed. An object carries {"type": ...}; an array carries
// one object per branch.
void CollectTypeNames(const ::tokenizer::Json& node, std::vector<std::string>& out) {
  if (node.is_object()) {
    if (auto type = ::tokenizer::GetString(node, "type")) {
      out.push_back(*type);
    }
  } else if (node.is_array()) {
    for (const auto& entry : node) CollectTypeNames(entry, out);
  }
}

absl::StatusOr<TokenizerMetadata> BuildMetadata(VendorTokenizer& first,
                                                const std::string& tokenizer_json) {
  TokenizerMetadata metadata;
  metadata.backend = kBackend;
  metadata.engine_version = kEngineVersion;

  metadata.total_vocab_size = first.VocabSize();

  ABSL_ASSIGN_OR_RETURN(auto vocab, first.VocabDump());
  if (!vocab.empty()) {
    metadata.max_token_id = vocab.back().first;
    metadata.contiguous_ids =
        vocab.front().first == 0 && vocab.back().first == static_cast<int64_t>(vocab.size()) - 1;
  }

  // Pipeline component names and byte-fallback, from the serialized
  // tokenizer the engine itself consumed.
  ABSL_ASSIGN_OR_RETURN(auto parsed, ::tokenizer::ParseJson(tokenizer_json, "tokenizer.json"));
  size_t added_tokens = 0;
  if (parsed.is_object()) {
    auto added = parsed.find("added_tokens");
    if (added != parsed.end() && added->is_array()) {
      added_tokens = added->size();
    }
    const struct {
      const char* key;
      std::vector<std::string>* out;
    } slots[] = {
        {"normalizer", &metadata.normalizer_types},
        {"pre_tokenizer", &metadata.pre_tokenizer_types},
        {"post_processor", &metadata.post_processor_types},
        {"decoder", &metadata.decoder_types},
    };
    for (const auto& slot : slots) {
      auto it = parsed.find(slot.key);
      if (it == parsed.end() || it->is_null()) continue;
      CollectTypeNames(*it, *slot.out);
    }
    auto model = parsed.find("model");
    if (model != parsed.end() && model->is_object()) {
      if (auto type = ::tokenizer::GetString(*model, "type")) {
        metadata.model_type = *type;
      }
      metadata.byte_fallback = ::tokenizer::GetBool(*model, "byte_fallback").value_or(false);
    }
  }
  metadata.base_vocab_size =
      metadata.total_vocab_size > added_tokens ? metadata.total_vocab_size - added_tokens : 0;

  if (auto bos = first.TryGetBosTokenId()) metadata.bos_id = TokenId(*bos);
  if (auto eos = first.TryGetEosTokenId()) metadata.eos_id = TokenId(*eos);
  if (auto pad = first.TryGetPadTokenId()) metadata.pad_id = TokenId(*pad);
  if (auto unk = first.TryGetUnkTokenId()) metadata.unk_id = TokenId(*unk);

  ABSL_ASSIGN_OR_RETURN(auto special, first.SpecialIds());
  metadata.special_ids = special;
  metadata.has_special_tokens = !special.empty();

  // Probe the serialized post-processor rather than trusting any config.
  {
    auto plain = first.Encode("", ToVendorEncode([] {
                                EncodeOptions options;
                                options.add_special_tokens = false;
                                return options;
                              }()));
    if (!plain.ok()) return AdaptStatus(plain.status());
    auto with_special = first.Encode("", ToVendorEncode([] {
                                       EncodeOptions options;
                                       options.add_special_tokens = true;
                                       return options;
                                     }()));
    if (!with_special.ok()) return AdaptStatus(with_special.status());
    metadata.default_adds_special_tokens = with_special->size() > plain->size();
  }

  metadata.model_max_length = first.ModelMaxLength();

  metadata.chat_template_names = first.ChatTemplateNames();
  std::sort(metadata.chat_template_names.begin(), metadata.chat_template_names.end());
  if (first.HasChatTemplate("default")) {
    metadata.default_chat_template = "default";
  }

  return metadata;
}

// ---------------------------------------------------------------------------
// Canonical metadata record
// ---------------------------------------------------------------------------

void AppendBytes(std::string& out, const std::string& value) {
  auto append_u32 = [&out](uint32_t v) {
    for (int shift = 24; shift >= 0; shift -= 8) {
      out.push_back(static_cast<char>((v >> shift) & 0xFF));
    }
  };
  append_u32(static_cast<uint32_t>(value.size()));
  out.append(value);
}

void AppendRecord(std::string& out, const std::string& tag, const std::string& value) {
  AppendBytes(out, tag);
  AppendBytes(out, value);
}

void AppendRecord(std::string& out, const std::string& tag, uint64_t value) {
  std::string bytes;
  for (int shift = 56; shift >= 0; shift -= 8) {
    bytes.push_back(static_cast<char>((value >> shift) & 0xFF));
  }
  AppendRecord(out, tag, bytes);
}

std::string JoinValues(const std::vector<std::string>& values) {
  std::string joined;
  for (const std::string& value : values) {
    if (!joined.empty()) joined.push_back('\n');
    joined.append(value);
  }
  return joined;
}

template <typename T>
std::string JoinNumbers(const std::vector<T>& values) {
  std::string joined;
  for (T value : values) {
    if (!joined.empty()) joined.push_back('\n');
    joined.append(std::to_string(value));
  }
  return joined;
}

// ---------------------------------------------------------------------------
// Streaming decoder
// ---------------------------------------------------------------------------

class StreamIncrementalDecoder : public IncrementalDecoder {
 public:
  StreamIncrementalDecoder(std::shared_ptr<SharedEngine> engine, bool skip_special_tokens)
      : engine_(std::move(engine)), skip_special_tokens_(skip_special_tokens) {}

  absl::StatusOr<std::optional<std::string>> Push(TokenId id) override {
    if (finished_) {
      return absl::FailedPreconditionError("incremental decoder already finished");
    }
    if (id.value() < 0) {
      return absl::InvalidArgumentError(absl::StrCat("token id ", id.value(), " is negative"));
    }
    if (stream_ == nullptr) {
      absl::Status create = absl::OkStatus();
      InstanceLease lease(engine_, create);
      if (!lease) return create;
      auto stream = lease->NewDecodeStream(skip_special_tokens_);
      if (!stream.ok()) return AdaptStatus(stream.status());
      stream_ = std::move(*stream);
    }
    absl::Status borrow = absl::OkStatus();
    InstanceLease lease(engine_, borrow);
    if (!lease) return borrow;
    auto chunk = lease->StepDecodeStream(*stream_, id.value());
    if (!chunk.ok()) {
      finished_ = true;
      return AdaptStatus(chunk.status());
    }
    if (chunk->has_value()) emitted_.append(**chunk);
    return std::move(*chunk);
  }

  absl::StatusOr<std::string> Finish() override {
    if (finished_) {
      return absl::FailedPreconditionError("incremental decoder already finished");
    }
    finished_ = true;
    if (stream_ == nullptr) return std::string();
    absl::Status borrow = absl::OkStatus();
    InstanceLease lease(engine_, borrow);
    if (!lease) return borrow;
    // Finish returns only the bytes the engine was still withholding, so
    // the concatenation of all chunks plus this result equals one-shot
    // decode.
    auto rest = lease->FinishDecodeStream(*stream_);
    if (!rest.ok()) return AdaptStatus(rest.status());
    emitted_.append(*rest);
    return *rest;
  }

  bool finished() const override { return finished_; }

 private:
  std::shared_ptr<SharedEngine> engine_;
  bool skip_special_tokens_;
  std::unique_ptr<::tokenizer::DecodeStream> stream_;
  std::string emitted_;
  bool finished_ = false;
};

// ---------------------------------------------------------------------------
// Facade
// ---------------------------------------------------------------------------

class QualifiedTokenizer final : public Tokenizer {
 public:
  explicit QualifiedTokenizer(std::shared_ptr<SharedEngine> engine) : engine_(std::move(engine)) {}

  absl::StatusOr<std::vector<TokenId>> Encode(std::string_view utf8,
                                              const EncodeOptions& options) const override {
    absl::Status borrow = absl::OkStatus();
    InstanceLease lease(engine_, borrow);
    if (!lease) return borrow;
    auto ids = lease->Encode(std::string(utf8), ToVendorEncode(options));
    if (!ids.ok()) return AdaptStatus(ids.status());
    std::vector<TokenId> out;
    out.reserve(ids->size());
    for (int32_t id : *ids) {
      if (id < 0) {
        return absl::InternalError("the engine emitted a negative token id");
      }
      out.push_back(TokenId(id));
    }
    return out;
  }

  absl::StatusOr<std::string> Decode(std::span<const TokenId> ids,
                                     const DecodeOptions& options) const override {
    for (TokenId id : ids) {
      if (id.value() < 0) {
        return absl::InvalidArgumentError(absl::StrCat("token id ", id.value(), " is negative"));
      }
    }
    absl::Status borrow = absl::OkStatus();
    InstanceLease lease(engine_, borrow);
    if (!lease) return borrow;
    ::tokenizer::DecodeOptions vendor;
    vendor.skip_special_tokens = options.skip_special_tokens;
    vendor.clean_up_tokenization_spaces = options.clean_up_tokenization_spaces;
    std::vector<int32_t> raw;
    raw.reserve(ids.size());
    for (TokenId id : ids) raw.push_back(id.value());
    auto text = lease->Decode(raw, vendor);
    if (!text.ok()) return AdaptStatus(text.status());
    return *text;
  }

  absl::StatusOr<std::unique_ptr<IncrementalDecoder>> NewIncrementalDecoder(
      const DecodeOptions& options) const override {
    // Streaming decode has cleanup disabled by contract:
    // cleanup can rewrite already-emitted bytes.
    return std::unique_ptr<IncrementalDecoder>(
        std::make_unique<StreamIncrementalDecoder>(engine_, options.skip_special_tokens));
  }

  absl::StatusOr<std::string> RenderChatPrompt(std::span<const ChatMessage> messages,
                                               const ChatRenderOptions& options) const override {
    std::vector<::tokenizer::ChatMessage> vendor_messages;
    vendor_messages.reserve(messages.size());
    for (const ChatMessage& message : messages) {
      ::tokenizer::ChatMessage converted;
      // The vendor role is a free-form string; the closed enum maps onto the
      // three supported roles.
      converted.role = message.role == ChatMessage::Role::kSystem
                           ? "system"
                           : (message.role == ChatMessage::Role::kAssistant ? "assistant" : "user");
      converted.content = message.content;
      vendor_messages.push_back(std::move(converted));
    }
    absl::Status borrow = absl::OkStatus();
    InstanceLease lease(engine_, borrow);
    if (!lease) return borrow;
    ::tokenizer::ChatTemplateOptions vendor;
    vendor.add_generation_prompt = options.add_generation_prompt;
    if (options.template_name.has_value()) {
      vendor.template_name = *options.template_name;
    }
    auto rendered = lease->ApplyChatTemplate(vendor_messages, vendor);
    if (!rendered.ok()) return AdaptStatus(rendered.status());
    return *rendered;
  }

  const TokenizerMetadata& metadata() const override { return engine_->metadata; }

 private:
  std::shared_ptr<SharedEngine> engine_;
};

}  // namespace

std::string CanonicalMetadataRecord(const TokenizerMetadata& metadata) {
  std::string out;
  out.append("inferx.tokenizer-metadata");
  const uint32_t schema = metadata.schema_version;
  for (int shift = 24; shift >= 0; shift -= 8) {
    out.push_back(static_cast<char>((schema >> shift) & 0xFF));
  }
  AppendRecord(out, "backend", metadata.backend);
  AppendRecord(out, "engine_version", metadata.engine_version);
  AppendRecord(out, "base_vocab_size", metadata.base_vocab_size);
  AppendRecord(out, "total_vocab_size", metadata.total_vocab_size);
  AppendRecord(out, "max_token_id", static_cast<uint64_t>(metadata.max_token_id));
  AppendRecord(out, "contiguous_ids", metadata.contiguous_ids ? "1" : "0");
  AppendRecord(out, "bos_id", metadata.bos_id ? std::to_string(metadata.bos_id->value()) : "-");
  AppendRecord(out, "eos_id", metadata.eos_id ? std::to_string(metadata.eos_id->value()) : "-");
  AppendRecord(out, "pad_id", metadata.pad_id ? std::to_string(metadata.pad_id->value()) : "-");
  AppendRecord(out, "unk_id", metadata.unk_id ? std::to_string(metadata.unk_id->value()) : "-");
  AppendRecord(out, "special_ids", JoinNumbers(metadata.special_ids));
  AppendRecord(out, "default_adds_special_tokens",
               metadata.default_adds_special_tokens ? "1" : "0");
  AppendRecord(out, "model_max_length", std::to_string(metadata.model_max_length));
  AppendRecord(out, "normalizer_types", JoinValues(metadata.normalizer_types));
  AppendRecord(out, "pre_tokenizer_types", JoinValues(metadata.pre_tokenizer_types));
  AppendRecord(out, "model_type", metadata.model_type);
  AppendRecord(out, "post_processor_types", JoinValues(metadata.post_processor_types));
  AppendRecord(out, "decoder_types", JoinValues(metadata.decoder_types));
  AppendRecord(out, "byte_fallback", metadata.byte_fallback ? "1" : "0");
  AppendRecord(out, "chat_template_names", JoinValues(metadata.chat_template_names));
  AppendRecord(out, "default_chat_template", metadata.default_chat_template.value_or(""));
  AppendRecord(out, "has_special_tokens", metadata.has_special_tokens ? "1" : "0");
  return out;
}

absl::StatusOr<std::unique_ptr<Tokenizer>> Tokenizer::Load(TokenizerArtifacts artifacts,
                                                           const TokenizerInstancePolicy& policy) {
  if (policy.instance_count < 1 || policy.instance_count > 32) {
    return absl::InvalidArgumentError("tokenizer instance count must be in 1..32");
  }

  auto engine = std::make_shared<SharedEngine>();

  for (uint32_t i = 0; i < policy.instance_count; ++i) {
    ::tokenizer::LocalArtifacts vendor_artifacts;
    // Every instance is constructed from the same bytes.
    vendor_artifacts.tokenizer_json = artifacts.tokenizer_json;
    vendor_artifacts.tokenizer_config = artifacts.tokenizer_config;
    vendor_artifacts.special_tokens_map = artifacts.special_tokens_map;
    vendor_artifacts.chat_template_jinja = artifacts.chat_template_jinja;
    vendor_artifacts.chat_template_json = artifacts.chat_template_json;
    vendor_artifacts.config = artifacts.model_config;

    ::tokenizer::PretrainedTokenizerOptions vendor_options;
    vendor_options.strict = policy.strict;

    auto instance = ::tokenizer::PretrainedTokenizer::FromLocalArtifacts(
        std::move(vendor_artifacts), vendor_options);
    if (!instance.ok()) return AdaptStatus(instance.status());
    engine->idle.push_back(std::move(*instance));
  }

  ABSL_ASSIGN_OR_RETURN(auto metadata,
                        BuildMetadata(*engine->idle.front(), artifacts.tokenizer_json));
  engine->metadata = std::move(metadata);
  engine->canonical_record = CanonicalMetadataRecord(engine->metadata);

  return std::unique_ptr<Tokenizer>(new QualifiedTokenizer(std::move(engine)));
}

}  // namespace inferx::tokenization
