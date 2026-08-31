#include "inferx/model/weight_planner.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace inferx::model {
namespace {

std::string NamesMessage(std::string_view prefix, const std::vector<std::string>& names) {
  std::string output(prefix);
  constexpr size_t kDiagnosticLimit = 16;
  const size_t shown = std::min(names.size(), kDiagnosticLimit);
  for (size_t i = 0; i < shown; ++i) {
    absl::StrAppend(&output, i == 0 ? ": " : ", ", names[i]);
  }
  if (shown < names.size()) {
    absl::StrAppend(&output, " (and ", names.size() - shown, " more)");
  }
  return output;
}

bool IsSupportedFloatingWeight(artifacts::ArtifactDType dtype) {
  return dtype == artifacts::ArtifactDType::kF16 || dtype == artifacts::ArtifactDType::kBf16 ||
         dtype == artifacts::ArtifactDType::kF32;
}

}  // namespace

absl::StatusOr<ExternalTensorCatalog> ExternalTensorCatalog::Build(
    std::vector<ExternalTensor> tensors) {
  for (const auto& tensor : tensors) {
    if (tensor.name.empty() || tensor.name != tensor.tensor.name) {
      return absl::InvalidArgumentError(
          "external tensor catalog name does not match header metadata");
    }
  }
  std::sort(
      tensors.begin(), tensors.end(),
      [](const ExternalTensor& lhs, const ExternalTensor& rhs) { return lhs.name < rhs.name; });
  for (size_t i = 1; i < tensors.size(); ++i) {
    if (tensors[i - 1].name == tensors[i].name) {
      return absl::DataLossError(
          absl::StrCat("external tensor occurs more than once: ", tensors[i].name));
    }
  }
  return ExternalTensorCatalog(std::move(tensors));
}

const ExternalTensor* ExternalTensorCatalog::Find(std::string_view name) const {
  const auto iterator = std::lower_bound(
      tensors_.begin(), tensors_.end(), name,
      [](const ExternalTensor& tensor, std::string_view target) { return tensor.name < target; });
  return iterator != tensors_.end() && iterator->name == name ? &*iterator : nullptr;
}

absl::StatusOr<WeightPlan> WeightPlanner::Build(const ModelSpec& model,
                                                const ParameterCatalog& parameters,
                                                const ExternalTensorCatalog& external) const {
  std::set<std::string> expected_names;
  std::set<std::string> consumed_names;
  std::vector<std::string> missing;
  std::optional<artifacts::ArtifactDType> package_dtype;
  WeightPlan plan;
  plan.coverage.expected = parameters.size();
  plan.coverage.external = external.tensors().size();

  for (const ParameterSpec& parameter : parameters) {
    if (!expected_names.insert(parameter.canonical_name).second) {
      return absl::InternalError("parameter catalog contains a duplicate canonical name");
    }
    const ExternalTensor* source = external.Find(parameter.canonical_name);
    if (parameter.alias_target.has_value()) {
      const ParameterId alias_target =
          parameter.alias_target.value_or(ParameterId(std::numeric_limits<uint32_t>::max()));
      const ParameterSpec* target = nullptr;
      for (const ParameterSpec& candidate : parameters) {
        if (candidate.id == alias_target) {
          target = &candidate;
          break;
        }
      }
      if (target == nullptr) {
        return absl::InternalError("parameter alias target does not exist");
      }
      const ExternalTensor* target_source = external.Find(target->canonical_name);
      if (target_source == nullptr) {
        missing.push_back(target->canonical_name);
        continue;
      }
      if (source != nullptr) {
        if (source->tensor.shape != parameter.shape ||
            source->tensor.dtype != target_source->tensor.dtype ||
            !IsSupportedFloatingWeight(source->tensor.dtype)) {
          return absl::FailedPreconditionError(
              "present tied lm_head.weight has incompatible shape or dtype");
        }
        if (!source->tensor_digest.has_value() || !target_source->tensor_digest.has_value() ||
            !(*source->tensor_digest == *target_source->tensor_digest)) {
          return absl::FailedPreconditionError(
              "tied lm_head.weight bytes differ from token embeddings");
        }
        consumed_names.insert(source->name);
      }
      consumed_names.insert(target_source->name);
      plan.aliases.push_back({parameter.id, alias_target});
      continue;
    }
    if (source == nullptr) {
      missing.push_back(parameter.canonical_name);
      continue;
    }
    consumed_names.insert(source->name);
    if (!IsSupportedFloatingWeight(source->tensor.dtype)) {
      return absl::UnimplementedError(
          absl::StrCat(parameter.canonical_name, ": source dtype ",
                       artifacts::ArtifactDTypeName(source->tensor.dtype),
                       " is not supported for dense Llama weights"));
    }
    if (std::find(parameter.allowed_source_dtypes.begin(), parameter.allowed_source_dtypes.end(),
                  source->tensor.dtype) == parameter.allowed_source_dtypes.end()) {
      return absl::UnimplementedError(
          absl::StrCat(parameter.canonical_name, ": source dtype rejected"));
    }
    if (source->tensor.shape != parameter.shape) {
      return absl::FailedPreconditionError(
          absl::StrCat(parameter.canonical_name, ": source shape mismatch"));
    }
    auto packed = artifacts::PackedTensorBytes(source->tensor.dtype, source->tensor.shape);
    if (!packed.ok()) return packed.status();
    if (*packed != source->tensor.packed_size_bytes) {
      return absl::DataLossError(
          absl::StrCat(parameter.canonical_name, ": source packed byte count mismatch"));
    }
    if (package_dtype.value_or(source->tensor.dtype) != source->tensor.dtype) {
      return absl::UnimplementedError("mixed source weight dtypes are unsupported in M3");
    }
    package_dtype = source->tensor.dtype;
    const artifacts::ArtifactByteRange file_range{source->tensor.absolute_file_offset,
                                                  source->tensor.packed_size_bytes};
    plan.items.push_back(
        {parameter.id,
         TensorSource{source->shard, source->file_identity, file_range, source->tensor.dtype,
                      source->tensor.shape, source->file_digest},
         TransformSpec{TransformKind::kIdentity, source->tensor.shape, parameter.shape},
         ShardKind::kReplicatedFullTensor});
  }

  if (!missing.empty()) {
    std::sort(missing.begin(), missing.end());
    missing.erase(std::unique(missing.begin(), missing.end()), missing.end());
    return absl::FailedPreconditionError(NamesMessage("missing required weights", missing));
  }
  std::vector<std::string> unexpected;
  for (const ExternalTensor& tensor : external.tensors()) {
    if (!consumed_names.contains(tensor.name)) unexpected.push_back(tensor.name);
  }
  if (!unexpected.empty()) {
    return absl::FailedPreconditionError(NamesMessage("unexpected external weights", unexpected));
  }
  std::sort(plan.items.begin(), plan.items.end(),
            [](const WeightPlanItem& lhs, const WeightPlanItem& rhs) {
              return lhs.parameter < rhs.parameter;
            });
  std::sort(plan.aliases.begin(), plan.aliases.end(),
            [](const ParameterAlias& lhs, const ParameterAlias& rhs) {
              return lhs.parameter < rhs.parameter;
            });
  plan.coverage.assigned = plan.items.size();
  plan.coverage.aliased = plan.aliases.size();
  if (plan.coverage.assigned + plan.coverage.aliased != plan.coverage.expected) {
    return absl::InternalError("weight coverage counts do not reconcile");
  }
  // The safetensors header remains authoritative; a stale torch_dtype hint is
  // retained as provenance but intentionally does not reject the package.
  static_cast<void>(model);
  return plan;
}

}  // namespace inferx::model
