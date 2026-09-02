// Immutable tokenizer capability metadata, derived
// from the same qualified engine resolution that encode uses -- never from a
// second parser that might disagree. Carries the canonical byte record used
// in the model fingerprint preimage.

#ifndef INFERX_TOKENIZATION_TOKENIZER_METADATA_H_
#define INFERX_TOKENIZATION_TOKENIZER_METADATA_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "inferx/base/id.h"

namespace inferx::tokenization {

// Sorted, immutable capability record. Field set is fixed by schema v1;
// adding a field requires a schema bump and a fingerprint note.
struct TokenizerMetadata {
  // Backend identity: "inferx-tokenizers-c" plus the exact engine version,
  // e.g. "hf-tokenizers-0.21.2".
  std::string backend;
  std::string engine_version;

  // Vocabulary geometry. `base_vocab_size` excludes added tokens;
  // `total_vocab_size` includes them.
  uint64_t base_vocab_size = 0;
  uint64_t total_vocab_size = 0;
  int64_t max_token_id = -1;  // -1 only for an empty vocabulary.
  // True when the emitted ids are exactly 0..total_vocab_size-1.
  bool contiguous_ids = false;

  std::optional<TokenId> bos_id;
  std::optional<TokenId> eos_id;
  std::optional<TokenId> pad_id;
  std::optional<TokenId> unk_id;
  std::vector<int32_t> special_ids;  // sorted, unique

  // What the serialized post-processor actually does (not what a config
  // claims): recorded by probing the engine with add_special_tokens.
  bool default_adds_special_tokens = false;

  // Advisory model maximum from tokenizer_config.json; kNoMaxLength when the
  // checkpoint declares none.
  int64_t model_max_length = -1;

  // Component type names present in the serialized pipeline, sorted.
  std::vector<std::string> normalizer_types;
  std::vector<std::string> pre_tokenizer_types;
  std::string model_type;
  std::vector<std::string> post_processor_types;
  std::vector<std::string> decoder_types;

  bool byte_fallback = false;

  // Chat templates the checkpoint ships, sorted; the selected default only
  // when the checkpoint explicitly declares one.
  std::vector<std::string> chat_template_names;
  std::optional<std::string> default_chat_template;

  // True when the engine's added vocabulary marks any token special; with
  // skip_special_tokens the engine drops exactly these.
  bool has_special_tokens = false;

  // Schema version of this metadata record.
  uint32_t schema_version = 1;
};

// The canonical metadata record for the fingerprint preimage: sorted,
// length-prefixed binary records, stable across runs and machines. Member
// order in the struct is irrelevant; the encoding is not.
std::string CanonicalMetadataRecord(const TokenizerMetadata& metadata);

}  // namespace inferx::tokenization

#endif  // INFERX_TOKENIZATION_TOKENIZER_METADATA_H_
