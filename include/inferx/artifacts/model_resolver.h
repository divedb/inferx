#pragma once

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"

namespace inferx::artifacts {

enum class ModelSource {
  kLocalDirectory,
  kHuggingFaceCache,
  kHuggingFaceDownload,
};

[[nodiscard]] std::string_view ModelSourceName(ModelSource source) noexcept;

struct ResolvedModel {
  std::filesystem::path path;
  ModelSource source = ModelSource::kLocalDirectory;
  std::string model;
  std::string revision;
};

// Mirrors the common vLLM/Hugging Face model-source controls. MODEL itself is
// either a local directory or a case-preserving Hub repository ID. A remote ID
// is looked up in local_model_dirs and the standard Hugging Face cache before
// any HTTP request is made.
struct ModelResolverOptions {
  std::string revision = "main";
  std::optional<std::filesystem::path> download_dir;
  std::vector<std::filesystem::path> local_model_dirs;
  std::optional<std::string> endpoint;
  std::optional<std::string> token;
  std::optional<std::string> proxy;
  std::chrono::milliseconds timeout{30'000};
  int max_retries = 2;
  bool local_files_only = false;
};

class ModelResolver {
 public:
  [[nodiscard]] absl::StatusOr<ResolvedModel> Resolve(
      std::string_view model, const ModelResolverOptions& options = {}) const;
};

}  // namespace inferx::artifacts
