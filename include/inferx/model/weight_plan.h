#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "inferx/artifacts/artifact_file.h"
#include "inferx/artifacts/artifact_tensor.h"
#include "inferx/artifacts/digest.h"
#include "inferx/artifacts/safe_relative_path.h"
#include "inferx/model/parameter_spec.h"

namespace inferx::model {

enum class TransformKind : uint8_t { kIdentity = 0 };
enum class ShardKind : uint8_t { kReplicatedFullTensor = 0 };

struct TransformSpec {
  TransformKind kind = TransformKind::kIdentity;
  artifacts::ArtifactShape input_shape;
  artifacts::ArtifactShape output_shape;
};

struct TensorSource {
  artifacts::SafeRelativePath shard;
  artifacts::FileIdentity expected_file;
  artifacts::ArtifactByteRange file_range;
  artifacts::ArtifactDtype dtype;
  artifacts::ArtifactShape shape;
  artifacts::Digest256 file_digest;
};

struct WeightPlanItem {
  ParameterId parameter;
  TensorSource source;
  TransformSpec transform;
  ShardKind logical_shard = ShardKind::kReplicatedFullTensor;
};

struct ParameterAlias {
  ParameterId parameter;
  ParameterId target;
};

struct WeightCoverageSummary {
  uint64_t expected = 0;
  uint64_t assigned = 0;
  uint64_t aliased = 0;
  uint64_t external = 0;
};

struct WeightPlan {
  static constexpr uint32_t kSchemaVersion = 1;
  uint32_t schema_version = kSchemaVersion;
  std::vector<WeightPlanItem> items;
  std::vector<ParameterAlias> aliases;
  WeightCoverageSummary coverage;
};

}  // namespace inferx::model
