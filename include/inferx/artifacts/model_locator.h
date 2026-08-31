#pragma once

#include <filesystem>
#include <memory>

#include "absl/status/statusor.h"
#include "inferx/artifacts/artifact_file.h"
#include "inferx/artifacts/artifact_limits.h"
#include "inferx/artifacts/safe_relative_path.h"

namespace inferx::artifacts {

class ArtifactSession {
 public:
  ~ArtifactSession();
  ArtifactSession(ArtifactSession&&) noexcept;
  ArtifactSession& operator=(ArtifactSession&&) noexcept;
  ArtifactSession(const ArtifactSession&) = delete;
  ArtifactSession& operator=(const ArtifactSession&) = delete;

  absl::StatusOr<ArtifactFile> OpenRegular(const SafeRelativePath& path) const;
  absl::StatusOr<bool> ExistsRegular(const SafeRelativePath& path) const;
  const std::filesystem::path& diagnostic_root() const { return root_; }
  const ArtifactLimits& limits() const { return limits_; }

 private:
  friend class ModelLocator;
  ArtifactSession(int root_fd, std::filesystem::path root, ArtifactLimits limits);

  int root_fd_ = -1;
  std::filesystem::path root_;
  ArtifactLimits limits_;
};

class ModelLocator {
 public:
  static absl::StatusOr<ArtifactSession> OpenLocal(const std::filesystem::path& root,
                                                   const ArtifactLimits& limits = {});
};

}  // namespace inferx::artifacts
