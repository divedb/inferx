#include "inferx/artifacts/model_fingerprint.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"

namespace inferx::artifacts {
namespace {

std::array<std::byte, 4> U32LittleEndian(uint32_t value) {
  std::array<std::byte, 4> output{};
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = static_cast<std::byte>((value >> (8 * i)) & 0xFFU);
  }
  return output;
}

std::array<std::byte, 8> U64LittleEndian(uint64_t value) {
  std::array<std::byte, 8> output{};
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = static_cast<std::byte>((value >> (8 * i)) & 0xFFU);
  }
  return output;
}

void Append(std::vector<std::byte>& target, std::string_view text) {
  const auto* begin = reinterpret_cast<const std::byte*>(text.data());
  target.insert(target.end(), begin, begin + text.size());
}

template <size_t N>
void Append(std::vector<std::byte>& target, const std::array<std::byte, N>& bytes) {
  target.insert(target.end(), bytes.begin(), bytes.end());
}

}  // namespace

absl::StatusOr<ModelFingerprint> ModelFingerprint::Build(ModelFingerprintInput input) {
  if (input.architecture.empty()) {
    return absl::InvalidArgumentError("model fingerprint architecture is empty");
  }
  if (input.semantic_config.empty()) {
    return absl::InvalidArgumentError("model fingerprint semantic config is empty");
  }
  if (input.model_schema_version == 0 || input.weight_plan_schema_version == 0) {
    return absl::InvalidArgumentError("model fingerprint component schema version is zero");
  }
  if (input.tokenizer_capability.empty()) {
    return absl::FailedPreconditionError(
        "model fingerprint requires qualified tokenizer capability metadata");
  }
  if (input.artifacts.empty()) {
    return absl::InvalidArgumentError("model fingerprint artifact set is empty");
  }
  if (input.source_layout.empty() || input.quantization.empty() || input.adapter.empty()) {
    return absl::InvalidArgumentError("model fingerprint compatibility records must not be empty");
  }
  if (input.rope.empty()) {
    return absl::InvalidArgumentError("model fingerprint RoPE policy is empty");
  }
  std::sort(input.artifacts.begin(), input.artifacts.end(),
            [](const FingerprintedArtifact& lhs, const FingerprintedArtifact& rhs) {
              return lhs.path < rhs.path;
            });
  for (size_t i = 1; i < input.artifacts.size(); ++i) {
    if (input.artifacts[i - 1].path == input.artifacts[i].path) {
      return absl::InvalidArgumentError("model fingerprint contains a duplicate artifact path");
    }
  }

  Hasher hasher;
  if (auto status = hasher.Update("inferx.model-fingerprint"); !status.ok()) {
    return status;
  }
  const auto schema = U32LittleEndian(kSchemaVersion);
  if (auto status = hasher.Update(schema); !status.ok()) return status;
  if (auto status = hasher.AddRecord("architecture", input.architecture); !status.ok()) {
    return status;
  }
  if (auto status = hasher.AddRecord("semantic_config", input.semantic_config); !status.ok()) {
    return status;
  }
  const auto model_schema = U32LittleEndian(input.model_schema_version);
  if (auto status = hasher.AddRecord("model_schema", model_schema); !status.ok()) {
    return status;
  }
  const auto plan_schema = U32LittleEndian(input.weight_plan_schema_version);
  if (auto status = hasher.AddRecord("weight_plan_schema", plan_schema); !status.ok()) {
    return status;
  }
  if (auto status = hasher.AddRecord("tokenizer_capability", input.tokenizer_capability);
      !status.ok()) {
    return status;
  }
  for (const auto& artifact : input.artifacts) {
    std::vector<std::byte> value;
    value.reserve(artifact.path.string().size() + 8 + Digest256::kSize);
    Append(value, artifact.path.string());
    Append(value, U64LittleEndian(artifact.size));
    value.insert(value.end(), artifact.digest.bytes().begin(), artifact.digest.bytes().end());
    if (auto status = hasher.AddRecord("artifact", value); !status.ok()) {
      return status;
    }
  }
  if (auto status = hasher.AddRecord("model_revision", input.model_revision); !status.ok()) {
    return status;
  }
  if (auto status = hasher.AddRecord("source_layout", input.source_layout); !status.ok()) {
    return status;
  }
  if (auto status = hasher.AddRecord("quantization", input.quantization); !status.ok()) {
    return status;
  }
  if (auto status = hasher.AddRecord("adapter", input.adapter); !status.ok()) {
    return status;
  }
  if (auto status = hasher.AddRecord("rope", input.rope); !status.ok()) {
    return status;
  }
  auto digest = hasher.Finalize();
  if (!digest.ok()) return digest.status();
  return ModelFingerprint(*digest);
}

}  // namespace inferx::artifacts
