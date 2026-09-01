#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace inferx::artifacts {

struct FileIdentity {
  uint64_t device = 0;
  uint64_t inode = 0;
  uint64_t size = 0;
  uint32_t mode = 0;
  int64_t modified_seconds = 0;
  int64_t modified_nanoseconds = 0;
  int64_t changed_seconds = 0;
  int64_t changed_nanoseconds = 0;
  uint64_t internal_id = 0;

  bool SameFileAndVersion(const FileIdentity& other) const;
};

class ArtifactFile {
 public:
  ArtifactFile() = default;
  ~ArtifactFile();
  ArtifactFile(ArtifactFile&& other) noexcept;
  ArtifactFile& operator=(ArtifactFile&& other) noexcept;
  ArtifactFile(const ArtifactFile&) = delete;
  ArtifactFile& operator=(const ArtifactFile&) = delete;

  const FileIdentity& identity() const { return identity_; }
  const std::string& relative_path() const { return relative_path_; }
  int native_fd() const { return fd_; }

  absl::Status ReadExact(uint64_t offset, std::span<std::byte> output) const;
  absl::StatusOr<std::vector<std::byte>> ReadAll(uint64_t limit) const;
  absl::StatusOr<ArtifactFile> Duplicate() const;
  absl::Status CheckUnchanged() const;

 private:
  friend class ArtifactSession;
  ArtifactFile(int fd, FileIdentity identity, std::string relative_path);

  int fd_ = -1;
  FileIdentity identity_;
  std::string relative_path_;
};

}  // namespace inferx::artifacts
