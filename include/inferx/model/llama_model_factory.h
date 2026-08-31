#pragma once

#include "absl/status/statusor.h"
#include "inferx/artifacts/artifact_file.h"
#include "inferx/artifacts/artifact_limits.h"
#include "inferx/model/model_spec.h"
#include "inferx/model/parameter_spec.h"

namespace inferx::model {

class LlamaModelFactory {
 public:
  absl::StatusOr<ModelSpec> ParseConfig(const artifacts::ArtifactFile& config,
                                        const artifacts::ArtifactLimits& limits = {}) const;
  absl::StatusOr<ParameterCatalog> BuildParameters(const ModelSpec& spec) const;
};

}  // namespace inferx::model
