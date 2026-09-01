#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace inferx::artifacts::internal {

class HuggingFaceCache {
 public:
  static std::filesystem::path ResolveRoot(
      const std::optional<std::filesystem::path>& override_root);

  HuggingFaceCache(std::filesystem::path root, std::string repo_id);

  [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }
  [[nodiscard]] const std::filesystem::path& repo_dir() const noexcept { return repo_dir_; }
  [[nodiscard]] std::filesystem::path SnapshotDir(std::string_view revision) const;
  [[nodiscard]] std::filesystem::path BlobPath(std::string_view etag) const;
  [[nodiscard]] std::filesystem::path RefPath(std::string_view revision) const;
  [[nodiscard]] std::filesystem::path InferXSnapshotDir(std::string_view revision) const;

  [[nodiscard]] std::optional<std::string> ReadRef(std::string_view revision) const;
  [[nodiscard]] absl::Status WriteRef(std::string_view revision, std::string_view sha) const;
  [[nodiscard]] absl::StatusOr<std::filesystem::path> EnsureBlobsDir() const;
  [[nodiscard]] absl::Status StoreFile(std::string_view revision, std::string_view filename,
                                       std::string_view etag,
                                       const std::filesystem::path& staged_file) const;
  [[nodiscard]] absl::StatusOr<std::filesystem::path> Materialize(std::string_view revision) const;

 private:
  std::filesystem::path root_;
  std::string repo_id_;
  std::filesystem::path repo_dir_;
};

[[nodiscard]] std::string HuggingFaceRepoDirName(std::string_view repo_id);
[[nodiscard]] bool IsFullCommitSha(std::string_view revision);
[[nodiscard]] bool SnapshotIsUsable(const std::filesystem::path& snapshot);

}  // namespace inferx::artifacts::internal
