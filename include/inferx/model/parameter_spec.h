#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "inferx/artifacts/artifact_tensor.h"

namespace inferx::model {

class ParameterId {
 public:
  explicit ParameterId(uint32_t value) : value_(value) {}
  uint32_t value() const { return value_; }
  friend bool operator==(ParameterId, ParameterId) = default;
  friend auto operator<=>(ParameterId, ParameterId) = default;

 private:
  uint32_t value_;
};

enum class ParameterRole : uint8_t {
  kTokenEmbedding,
  kAttentionNorm,
  kAttentionQuery,
  kAttentionKey,
  kAttentionValue,
  kAttentionOutput,
  kMlpNorm,
  kMlpGate,
  kMlpUp,
  kMlpDown,
  kFinalNorm,
  kLmHead,
};

struct ParameterSpec {
  ParameterId id;
  std::string canonical_name;
  ParameterRole role;
  artifacts::ArtifactShape shape;
  std::vector<artifacts::ArtifactDType> allowed_source_dtypes;
  std::optional<uint32_t> logical_layer;
  std::optional<ParameterId> alias_target;
};

using ParameterCatalog = std::vector<ParameterSpec>;

}  // namespace inferx::model
