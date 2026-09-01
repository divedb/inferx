#include "inferx/artifacts/artifact_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace inferx::artifacts {
namespace {

std::atomic<uint64_t> g_next_file_id{1};

absl::Status ErrnoStatus(absl::StatusCode code, std::string_view operation, int error) {
  return absl::Status(code, absl::StrCat(operation, " failed (errno=", error, ")"));
}

absl::StatusOr<uint64_t> NextFileId() {
  uint64_t current = g_next_file_id.load(std::memory_order_relaxed);
  while (current != 0) {
    const uint64_t next = current == std::numeric_limits<uint64_t>::max() ? 0 : current + 1;
    if (g_next_file_id.compare_exchange_weak(current, next, std::memory_order_relaxed,
                                             std::memory_order_relaxed)) {
      return current;
    }
  }
  return absl::ResourceExhaustedError("artifact file ID space exhausted");
}

absl::StatusOr<FileIdentity> IdentityFromFd(int fd) {
  struct stat info {};
  while (::fstat(fd, &info) != 0) {
    if (errno == EINTR) continue;
    return ErrnoStatus(absl::StatusCode::kInternal, "fstat", errno);
  }
  if (info.st_size < 0) {
    return absl::DataLossError("regular file reported a negative size");
  }
  FileIdentity identity;
  identity.device = static_cast<uint64_t>(info.st_dev);
  identity.inode = static_cast<uint64_t>(info.st_ino);
  identity.size = static_cast<uint64_t>(info.st_size);
  identity.mode = static_cast<uint32_t>(info.st_mode);
  identity.modified_seconds = static_cast<int64_t>(info.st_mtim.tv_sec);
  identity.modified_nanoseconds = static_cast<int64_t>(info.st_mtim.tv_nsec);
  identity.changed_seconds = static_cast<int64_t>(info.st_ctim.tv_sec);
  identity.changed_nanoseconds = static_cast<int64_t>(info.st_ctim.tv_nsec);
  return identity;
}

}  // namespace

bool FileIdentity::SameFileAndVersion(const FileIdentity& other) const {
  return device == other.device && inode == other.inode && size == other.size &&
         mode == other.mode && modified_seconds == other.modified_seconds &&
         modified_nanoseconds == other.modified_nanoseconds &&
         changed_seconds == other.changed_seconds &&
         changed_nanoseconds == other.changed_nanoseconds;
}

ArtifactFile::ArtifactFile(int fd, FileIdentity identity, std::string relative_path)
    : fd_(fd), identity_(identity), relative_path_(std::move(relative_path)) {}

ArtifactFile::~ArtifactFile() {
  if (fd_ >= 0) ::close(fd_);
}

ArtifactFile::ArtifactFile(ArtifactFile&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      identity_(other.identity_),
      relative_path_(std::move(other.relative_path_)) {}

ArtifactFile& ArtifactFile::operator=(ArtifactFile&& other) noexcept {
  if (this == &other) return *this;
  if (fd_ >= 0) ::close(fd_);
  fd_ = std::exchange(other.fd_, -1);
  identity_ = other.identity_;
  relative_path_ = std::move(other.relative_path_);
  return *this;
}

absl::Status ArtifactFile::ReadExact(uint64_t offset, std::span<std::byte> output) const {
  if (fd_ < 0) return absl::FailedPreconditionError("artifact file is closed");
  if (offset > identity_.size || output.size() > identity_.size - offset) {
    return absl::DataLossError("artifact read exceeds the opened file size");
  }
  if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
    return absl::OutOfRangeError("artifact offset does not fit off_t");
  }

  size_t completed = 0;
  while (completed < output.size()) {
    const uint64_t position = offset + completed;
    if (position > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
      return absl::OutOfRangeError("artifact read position does not fit off_t");
    }
    const size_t remaining = output.size() - completed;
    const size_t request = remaining > static_cast<size_t>(std::numeric_limits<ssize_t>::max())
                               ? static_cast<size_t>(std::numeric_limits<ssize_t>::max())
                               : remaining;
    const ssize_t count =
        ::pread(fd_, output.data() + completed, request, static_cast<off_t>(position));
    if (count < 0) {
      if (errno == EINTR) continue;
      return ErrnoStatus(absl::StatusCode::kInternal, "pread", errno);
    }
    if (count == 0) {
      return absl::DataLossError("unexpected EOF while reading artifact");
    }
    completed += static_cast<size_t>(count);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::byte>> ArtifactFile::ReadAll(uint64_t limit) const {
  if (identity_.size > limit) {
    return absl::ResourceExhaustedError("artifact exceeds the configured read limit");
  }
  if (identity_.size > std::numeric_limits<size_t>::max()) {
    return absl::ResourceExhaustedError("artifact cannot be represented in host address space");
  }
  std::vector<std::byte> result(static_cast<size_t>(identity_.size));
  absl::Status status = ReadExact(0, result);
  if (!status.ok()) return status;
  return result;
}

absl::StatusOr<ArtifactFile> ArtifactFile::Duplicate() const {
  if (fd_ < 0) return absl::FailedPreconditionError("artifact file is closed");
  if (auto status = CheckUnchanged(); !status.ok()) return status;
  int duplicate = -1;
  do {
    duplicate = ::fcntl(fd_, F_DUPFD_CLOEXEC, 0);
  } while (duplicate < 0 && errno == EINTR);
  if (duplicate < 0) {
    return ErrnoStatus(absl::StatusCode::kInternal, "artifact fd duplication", errno);
  }
  ArtifactFile result(duplicate, identity_, relative_path_);
  if (auto status = result.CheckUnchanged(); !status.ok()) return status;
  return result;
}

absl::Status ArtifactFile::CheckUnchanged() const {
  if (fd_ < 0) return absl::FailedPreconditionError("artifact file is closed");
  auto current = IdentityFromFd(fd_);
  if (!current.ok()) return current.status();
  if (!identity_.SameFileAndVersion(*current)) {
    return absl::AbortedError("artifact changed while it was being consumed");
  }
  return absl::OkStatus();
}

// Used only by the rooted opener, which must snapshot identity immediately
// after the fd is acquired.
absl::StatusOr<FileIdentity> SnapshotFileIdentityForArtifact(int fd) {
  auto identity = IdentityFromFd(fd);
  if (!identity.ok()) return identity.status();
  auto internal_id = NextFileId();
  if (!internal_id.ok()) return internal_id.status();
  identity->internal_id = *internal_id;
  return identity;
}

}  // namespace inferx::artifacts
