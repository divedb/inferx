#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/artifacts/artifact_file.h"
#include "inferx/artifacts/artifact_tensor.h"
#include "inferx/artifacts/digest.h"
#include "inferx/artifacts/safe_relative_path.h"
#include "inferx/model/model_spec.h"
#include "inferx/model/parameter_spec.h"
#include "inferx/model/weight_plan.h"

namespace inferx::model {

struct ExternalTensor {
  std::string name;
  artifacts::SafeRelativePath shard;
  artifacts::FileIdentity file_identity;
  artifacts::ArtifactTensor tensor;
  artifacts::Digest256 file_digest;
  std::optional<artifacts::Digest256> tensor_digest;
};

class ExternalTensorCatalog {
 public:
  static absl::StatusOr<ExternalTensorCatalog> Build(std::vector<ExternalTensor> tensors);
  const ExternalTensor* Find(std::string_view name) const;
  const std::vector<ExternalTensor>& tensors() const { return tensors_; }

 private:
  explicit ExternalTensorCatalog(std::vector<ExternalTensor> tensors)
      : tensors_(std::move(tensors)) {}
  std::vector<ExternalTensor> tensors_;
};

class WeightPlanner {
 public:
  absl::StatusOr<WeightPlan> Build(const ModelSpec& model, const ParameterCatalog& parameters,
                                   const ExternalTensorCatalog& external) const;
};

}  // namespace inferx::model
