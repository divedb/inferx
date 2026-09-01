#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "inferx/artifacts/artifact_tensor.h"

namespace inferx::model {

struct LlamaSpec {
  uint64_t vocab_size = 0;
  uint64_t hidden_size = 0;
  uint64_t intermediate_size = 0;
  uint32_t num_hidden_layers = 0;
  uint32_t num_attention_heads = 0;
  uint32_t num_key_value_heads = 0;
  uint32_t head_dim = 0;
  uint64_t max_position_embeddings = 0;
  double rms_norm_eps = 0;
  double rope_theta = 10'000.0;
  double attention_dropout = 0;
  bool tie_word_embeddings = false;
  std::optional<int32_t> bos_token_id;
  std::vector<int32_t> eos_token_ids;
  std::optional<int32_t> pad_token_id;
  std::optional<artifacts::ArtifactDType> source_weight_type_hint;
};

class ModelSpec {
 public:
  static constexpr uint32_t kSchemaVersion = 1;

  explicit ModelSpec(LlamaSpec llama);
  const LlamaSpec& llama() const { return llama_; }
  std::span<const std::byte> canonical_bytes() const { return canonical_; }

 private:
  LlamaSpec llama_;
  std::vector<std::byte> canonical_;
};

}  // namespace inferx::model
