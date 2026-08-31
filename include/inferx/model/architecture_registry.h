#pragma once

#include <map>
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/model/llama_model_factory.h"

namespace inferx::model {

class ArchitectureRegistry {
 public:
  absl::Status Register(std::string model_type, std::unique_ptr<const LlamaModelFactory> factory);
  absl::StatusOr<const LlamaModelFactory*> Find(std::string_view model_type) const;

 private:
  std::map<std::string, std::unique_ptr<const LlamaModelFactory>, std::less<>> factories_;
};

}  // namespace inferx::model
