#include "inferx/artifacts/digest.h"

#include <blake3.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "inferx/artifacts/artifact_file.h"

namespace inferx::artifacts {
namespace {

constexpr char kHexDigits[] = "0123456789abcdef";

absl::Status UpdateU64(Hasher& hasher, uint64_t value) {
  std::array<std::byte, 8> encoded{};
  for (size_t i = 0; i < encoded.size(); ++i) {
    encoded[i] = static_cast<std::byte>((value >> (8 * i)) & 0xFFU);
  }
  return hasher.Update(encoded);
}

absl::StatusOr<uint8_t> ParseNibble(char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<uint8_t>(10 + value - 'a');
  }
  return absl::InvalidArgumentError("digest must contain lowercase hexadecimal characters");
}

}  // namespace

absl::StatusOr<Digest256> Digest256::ParseHex(std::string_view text) {
  if (text.size() != kSize * 2) {
    return absl::InvalidArgumentError("BLAKE3-256 digest must contain exactly 64 hex characters");
  }
  std::array<std::byte, kSize> bytes{};
  for (size_t i = 0; i < kSize; ++i) {
    auto high = ParseNibble(text[i * 2]);
    if (!high.ok()) return high.status();
    auto low = ParseNibble(text[i * 2 + 1]);
    if (!low.ok()) return low.status();
    bytes[i] = static_cast<std::byte>((*high << 4U) | *low);
  }
  return Digest256(bytes);
}

std::string Digest256::Hex() const {
  std::string result(kSize * 2, '0');
  for (size_t i = 0; i < kSize; ++i) {
    const uint8_t value = std::to_integer<uint8_t>(bytes_[i]);
    result[i * 2] = kHexDigits[value >> 4U];
    result[i * 2 + 1] = kHexDigits[value & 0x0FU];
  }
  return result;
}

bool operator==(const Digest256& lhs, const Digest256& rhs) {
  uint8_t difference = 0;
  for (size_t i = 0; i < Digest256::kSize; ++i) {
    difference |= std::to_integer<uint8_t>(lhs.bytes_[i] ^ rhs.bytes_[i]);
  }
  return difference == 0;
}

struct Hasher::Impl {
  blake3_hasher state{};
  bool finalized = false;
};

Hasher::Hasher() : impl_(std::make_unique<Impl>()) { blake3_hasher_init(&impl_->state); }

Hasher::~Hasher() = default;
Hasher::Hasher(Hasher&&) noexcept = default;
Hasher& Hasher::operator=(Hasher&&) noexcept = default;

absl::Status Hasher::Update(std::span<const std::byte> bytes) {
  if (!impl_ || impl_->finalized) {
    return absl::FailedPreconditionError("BLAKE3 hasher is finalized");
  }
  blake3_hasher_update(&impl_->state, bytes.data(), bytes.size());
  return absl::OkStatus();
}

absl::Status Hasher::Update(std::string_view text) {
  return Update(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(text.data()), text.size()));
}

absl::Status Hasher::AddRecord(std::string_view tag, std::span<const std::byte> value) {
  if (auto status = UpdateU64(*this, tag.size()); !status.ok()) return status;
  if (auto status = Update(tag); !status.ok()) return status;
  if (auto status = UpdateU64(*this, value.size()); !status.ok()) return status;
  return Update(value);
}

absl::Status Hasher::AddRecord(std::string_view tag, std::string_view value) {
  return AddRecord(tag, std::span<const std::byte>(reinterpret_cast<const std::byte*>(value.data()),
                                                   value.size()));
}

absl::StatusOr<Digest256> Hasher::Finalize() {
  if (!impl_ || impl_->finalized) {
    return absl::FailedPreconditionError("BLAKE3 hasher is finalized");
  }
  std::array<std::byte, Digest256::kSize> bytes{};
  blake3_hasher_finalize(&impl_->state, reinterpret_cast<uint8_t*>(bytes.data()), bytes.size());
  impl_->finalized = true;
  return Digest256(bytes);
}

absl::StatusOr<Digest256> HashBytes(std::span<const std::byte> bytes) {
  Hasher hasher;
  if (auto status = hasher.Update(bytes); !status.ok()) return status;
  return hasher.Finalize();
}

absl::StatusOr<Digest256> HashFileRange(const ArtifactFile& file, uint64_t offset, uint64_t size) {
  if (offset > file.identity().size || size > file.identity().size - offset) {
    return absl::InvalidArgumentError("hash range exceeds artifact file");
  }
  if (auto status = file.CheckUnchanged(); !status.ok()) return status;

  constexpr size_t kChunkSize = size_t{1024} * 1024;
  std::vector<std::byte> buffer(kChunkSize);
  Hasher hasher;
  uint64_t completed = 0;
  while (completed < size) {
    const uint64_t remaining = size - completed;
    const size_t count =
        static_cast<size_t>(remaining < kChunkSize ? remaining : static_cast<uint64_t>(kChunkSize));
    std::span<std::byte> chunk(buffer.data(), count);
    if (auto status = file.ReadExact(offset + completed, chunk); !status.ok()) {
      return status;
    }
    if (auto status = hasher.Update(chunk); !status.ok()) return status;
    completed += count;
  }
  if (auto status = file.CheckUnchanged(); !status.ok()) return status;
  return hasher.Finalize();
}

absl::StatusOr<Digest256> HashFile(const ArtifactFile& file) {
  return HashFileRange(file, 0, file.identity().size);
}

}  // namespace inferx::artifacts
