#pragma once

#include <compare>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"

namespace inferx::artifacts {

class SafeRelativePath {
 public:
  static absl::StatusOr<SafeRelativePath> Parse(std::string_view path);

  const std::string& string() const { return value_; }
  friend bool operator==(const SafeRelativePath&, const SafeRelativePath&) = default;
  friend auto operator<=>(const SafeRelativePath&, const SafeRelativePath&) = default;

 private:
  explicit SafeRelativePath(std::string value) : value_(std::move(value)) {}
  std::string value_;
};

}  // namespace inferx::artifacts
