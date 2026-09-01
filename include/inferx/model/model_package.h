#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/artifacts/artifact_limits.h"
#include "inferx/artifacts/model_fingerprint.h"
#include "inferx/artifacts/safe_relative_path.h"
#include "inferx/model/model_spec.h"
#include "inferx/model/parameter_spec.h"
#include "inferx/model/weight_plan.h"

namespace inferx::model {

// The artifact/model half of a package.  This value is deliberately distinct
// from the milestone's ValidatedModelPackage: it cannot be promoted until the
// tokenizer backend passes ADR 0025's qualification gate.
struct InspectedModelArtifacts {
  ModelSpec model_spec;
  ParameterCatalog parameters;
  WeightPlan weight_plan;
  std::vector<artifacts::FingerprintedArtifact> artifacts;
  artifacts::SafeRelativePath weights_entry;
  bool integrity_manifest = false;
  std::string model_revision;
};

class ModelArtifactLoader {
 public:
  absl::StatusOr<InspectedModelArtifacts> Inspect(
      const std::filesystem::path& root, const artifacts::ArtifactLimits& limits = {}) const;
};

}  // namespace inferx::model
