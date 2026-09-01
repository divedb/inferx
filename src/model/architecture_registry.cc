#include "inferx/model/architecture_registry.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace inferx::model {

absl::Status ArchitectureRegistry::Register(std::string model_type,
                                            std::unique_ptr<const LlamaModelFactory> factory) {
  if (model_type.empty() || factory == nullptr) {
    return absl::InvalidArgumentError("architecture registration requires a tag and factory");
  }
  if (factories_.contains(model_type)) {
    return absl::AlreadyExistsError(absl::StrCat("duplicate architecture tag: ", model_type));
  }
  factories_.emplace(std::move(model_type), std::move(factory));
  return absl::OkStatus();
}

absl::StatusOr<const LlamaModelFactory*> ArchitectureRegistry::Find(
    std::string_view model_type) const {
  const auto iterator = factories_.find(model_type);
  if (iterator == factories_.end()) {
    return absl::UnimplementedError(absl::StrCat("unsupported model_type: ", model_type));
  }
  return iterator->second.get();
}

}  // namespace inferx::model
