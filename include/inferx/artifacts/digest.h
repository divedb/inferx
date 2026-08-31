#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace inferx::artifacts {

class ArtifactFile;

class Digest256 {
 public:
  static constexpr size_t kSize = 32;

  Digest256() = default;
  explicit Digest256(std::array<std::byte, kSize> bytes) : bytes_(bytes) {}

  static absl::StatusOr<Digest256> ParseHex(std::string_view text);
  std::string Hex() const;
  std::span<const std::byte, kSize> bytes() const { return bytes_; }
  friend bool operator==(const Digest256& lhs, const Digest256& rhs);

 private:
  std::array<std::byte, kSize> bytes_{};
};

class Hasher {
 public:
  Hasher();
  ~Hasher();
  Hasher(Hasher&&) noexcept;
  Hasher& operator=(Hasher&&) noexcept;
  Hasher(const Hasher&) = delete;
  Hasher& operator=(const Hasher&) = delete;

  absl::Status Update(std::span<const std::byte> bytes);
  absl::Status Update(std::string_view text);
  absl::Status AddRecord(std::string_view tag, std::span<const std::byte> value);
  absl::Status AddRecord(std::string_view tag, std::string_view value);
  absl::StatusOr<Digest256> Finalize();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

absl::StatusOr<Digest256> HashBytes(std::span<const std::byte> bytes);
absl::StatusOr<Digest256> HashFile(const ArtifactFile& file);
absl::StatusOr<Digest256> HashFileRange(const ArtifactFile& file, uint64_t offset, uint64_t size);

}  // namespace inferx::artifacts
