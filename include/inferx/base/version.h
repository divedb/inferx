// InferX base version API. It must remain self-contained (it is validated by
// tools/check_headers.py) and includable from any translation unit in any
// order.

#ifndef INFERX_BASE_VERSION_H_
#define INFERX_BASE_VERSION_H_

#include <string_view>

namespace inferx {

// Compile-time InferX version, mirroring the CMake project version.
struct Version {
  int major;
  int minor;
  int patch;
};

// Runtime copy of the version the library was built from.
[[nodiscard]] Version GetVersion() noexcept;

// Stable dotted form, e.g. "0.0.0". The view has static storage duration.
[[nodiscard]] std::string_view GetVersionString() noexcept;

}  // namespace inferx

#endif  // INFERX_BASE_VERSION_H_
