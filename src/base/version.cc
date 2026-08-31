#include "inferx/base/version.h"

// INFERX_VERSION_MAJOR/MINOR/PATCH and INFERX_VERSION_STRING come from the
// target's compile definitions (mirroring the CMake project version); no
// generated headers, paths, timestamps, or usernames are embedded.

namespace inferx {

Version GetVersion() noexcept {
  return Version{
      .major = INFERX_VERSION_MAJOR,
      .minor = INFERX_VERSION_MINOR,
      .patch = INFERX_VERSION_PATCH,
  };
}

std::string_view GetVersionString() noexcept {
  static constexpr char kVersion[] = INFERX_VERSION_STRING;
  return kVersion;
}

}  // namespace inferx
