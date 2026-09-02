#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "inferx/artifacts/model_locator.h"

#if defined(__linux__)
#include <linux/openat2.h>
#endif

#include <cerrno>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace inferx::artifacts {

absl::StatusOr<FileIdentity> SnapshotFileIdentityForArtifact(int fd);

namespace {

absl::Status OpenError(std::string_view operation, int error) {
  absl::StatusCode code = absl::StatusCode::kInternal;
  if (error == ENOENT) code = absl::StatusCode::kNotFound;
  if (error == ELOOP || error == EXDEV || error == ENOTDIR) {
    code = absl::StatusCode::kPermissionDenied;
  }
  if (error == EACCES || error == EPERM) {
    code = absl::StatusCode::kPermissionDenied;
  }
  return absl::Status(code, absl::StrCat(operation, " failed (errno=", error, ")"));
}

int OpenWithFallback(int root_fd, std::string_view path, int* error) {
  int parent = -1;
  do {
    parent = ::fcntl(root_fd, F_DUPFD_CLOEXEC, 0);
  } while (parent < 0 && errno == EINTR);
  if (parent < 0) {
    *error = errno;
    return -1;
  }

  size_t begin = 0;
  while (true) {
    const size_t separator = path.find('/', begin);
    const bool last = separator == std::string_view::npos;
    const std::string component(
        path.substr(begin, last ? std::string_view::npos : separator - begin));
    const int flags =
        last ? O_RDONLY | O_CLOEXEC | O_NOFOLLOW : O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECTORY;
    int next = -1;
    do {
      next = ::openat(parent, component.c_str(), flags);
    } while (next < 0 && errno == EINTR);
    if (next < 0) {
      *error = errno;
      ::close(parent);
      return -1;
    }
    ::close(parent);
    parent = next;
    if (last) return parent;
    begin = separator + 1;
  }
}

int OpenBeneath(int root_fd, std::string_view path, int* error) {
#if defined(__linux__) && defined(SYS_openat2) && INFERX_USE_OPENAT2
  struct open_how how {};
  how.flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW;
  how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS;
  std::string terminated(path);
  int result = -1;
  do {
    result =
        static_cast<int>(::syscall(SYS_openat2, root_fd, terminated.c_str(), &how, sizeof(how)));
  } while (result < 0 && errno == EINTR);
  if (result >= 0) return result;
  if (errno != ENOSYS && errno != EINVAL && errno != E2BIG) {
    *error = errno;
    return -1;
  }
#endif
  return OpenWithFallback(root_fd, path, error);
}

}  // namespace

ArtifactSession::ArtifactSession(int root_fd, std::filesystem::path root, ArtifactLimits limits)
    : root_fd_(root_fd), root_(std::move(root)), limits_(limits) {}

ArtifactSession::~ArtifactSession() {
  if (root_fd_ >= 0) ::close(root_fd_);
}

ArtifactSession::ArtifactSession(ArtifactSession&& other) noexcept
    : root_fd_(std::exchange(other.root_fd_, -1)),
      root_(std::move(other.root_)),
      limits_(other.limits_) {}

ArtifactSession& ArtifactSession::operator=(ArtifactSession&& other) noexcept {
  if (this == &other) return *this;
  if (root_fd_ >= 0) ::close(root_fd_);
  root_fd_ = std::exchange(other.root_fd_, -1);
  root_ = std::move(other.root_);
  limits_ = other.limits_;
  return *this;
}

absl::StatusOr<ArtifactFile> ArtifactSession::OpenRegular(const SafeRelativePath& path) const {
  if (root_fd_ < 0) {
    return absl::FailedPreconditionError("artifact session is closed");
  }
  int error = 0;
  const int fd = OpenBeneath(root_fd_, path.string(), &error);
  if (fd < 0) return OpenError("rooted artifact open", error);

  auto identity = SnapshotFileIdentityForArtifact(fd);
  if (!identity.ok()) {
    ::close(fd);
    return identity.status();
  }
  if (!S_ISREG(identity->mode)) {
    ::close(fd);
    return absl::InvalidArgumentError("artifact path does not identify a regular file");
  }
  return ArtifactFile(fd, *identity, path.string());
}

absl::StatusOr<bool> ArtifactSession::ExistsRegular(const SafeRelativePath& path) const {
  auto file = OpenRegular(path);
  if (file.ok()) return true;
  if (absl::IsNotFound(file.status())) return false;
  return file.status();
}

absl::StatusOr<ArtifactSession> ModelLocator::OpenLocal(const std::filesystem::path& root,
                                                        const ArtifactLimits& limits) {
  absl::Status limits_status = limits.Validate();
  if (!limits_status.ok()) return limits_status;

  std::error_code error;
  const std::filesystem::path canonical = std::filesystem::canonical(root, error);
  if (error) {
    return absl::NotFoundError(absl::StrCat("cannot resolve model root: ", error.message()));
  }
  int fd = -1;
  do {
    fd = ::open(canonical.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) return OpenError("model root open", errno);
  return ArtifactSession(fd, canonical, limits);
}

}  // namespace inferx::artifacts
