#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/artifacts/digest.h"
#include "inferx/artifacts/safe_relative_path.h"

namespace inferx::artifacts {

struct FingerprintedArtifact {
  SafeRelativePath path;
  uint64_t size = 0;
  Digest256 digest;
};

struct ModelFingerprintInput {
  std::string architecture;
  std::vector<std::byte> semantic_config;
  uint32_t model_schema_version = 1;
  uint32_t weight_plan_schema_version = 1;
  std::vector<std::byte> tokenizer_capability;
  std::vector<FingerprintedArtifact> artifacts;
  std::string model_revision;
  std::string source_layout = "hf-safetensors-row-major-v1";
  std::string quantization = "none";
  std::string adapter = "none";
  std::vector<std::byte> rope;
};

class ModelFingerprint {
 public:
  static constexpr uint32_t kSchemaVersion = 1;

  static absl::StatusOr<ModelFingerprint> Build(ModelFingerprintInput input);
  uint32_t schema_version() const { return kSchemaVersion; }
  std::string_view algorithm() const { return "blake3-256"; }
  const Digest256& digest() const { return digest_; }

 private:
  explicit ModelFingerprint(Digest256 digest) : digest_(digest) {}
  Digest256 digest_;
};

}  // namespace inferx::artifacts
