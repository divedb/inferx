#include "inferx/artifacts/safe_relative_path.h"

#include <cstddef>
#include <string>
#include <string_view>

#include "absl/status/status.h"

namespace inferx::artifacts {
namespace {

bool IsValidUtf8(std::string_view text) {
  const auto* data = reinterpret_cast<const unsigned char*>(text.data());
  size_t index = 0;
  while (index < text.size()) {
    const unsigned char first = data[index++];
    if (first < 0x80) continue;
    uint32_t codepoint = 0;
    size_t continuation = 0;
    if ((first & 0xE0U) == 0xC0U) {
      codepoint = first & 0x1FU;
      continuation = 1;
    } else if ((first & 0xF0U) == 0xE0U) {
      codepoint = first & 0x0FU;
      continuation = 2;
    } else if ((first & 0xF8U) == 0xF0U) {
      codepoint = first & 0x07U;
      continuation = 3;
    } else {
      return false;
    }
    if (continuation > text.size() - index) return false;
    for (size_t i = 0; i < continuation; ++i) {
      const unsigned char next = data[index++];
      if ((next & 0xC0U) != 0x80U) return false;
      codepoint = (codepoint << 6U) | (next & 0x3FU);
    }
    if ((continuation == 1 && codepoint < 0x80U) || (continuation == 2 && codepoint < 0x800U) ||
        (continuation == 3 && codepoint < 0x10000U) || codepoint > 0x10FFFFU ||
        (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::StatusOr<SafeRelativePath> SafeRelativePath::Parse(std::string_view path) {
  if (path.empty()) {
    return absl::InvalidArgumentError("artifact path is empty");
  }
  if (!IsValidUtf8(path)) {
    return absl::InvalidArgumentError("artifact path is not valid UTF-8");
  }
  if (path.front() == '/' || path.find('\\') != std::string_view::npos ||
      path.find('\0') != std::string_view::npos) {
    return absl::InvalidArgumentError("artifact path must be a normalized relative slash path");
  }

  size_t begin = 0;
  while (begin <= path.size()) {
    const size_t end = path.find('/', begin);
    const std::string_view component =
        path.substr(begin, end == std::string_view::npos ? std::string_view::npos : end - begin);
    if (component.empty() || component == "." || component == "..") {
      return absl::InvalidArgumentError(
          "artifact path contains an empty, dot, or dot-dot component");
    }
    if (component.find(':') != std::string_view::npos && begin == 0) {
      return absl::InvalidArgumentError("artifact path contains a root-name prefix");
    }
    if (end == std::string_view::npos) break;
    begin = end + 1;
  }
  return SafeRelativePath(std::string(path));
}

}  // namespace inferx::artifacts
