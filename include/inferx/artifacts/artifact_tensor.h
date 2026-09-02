#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"

namespace inferx::artifacts {

struct ArtifactByteRange {
  uint64_t offset = 0;
  uint64_t size = 0;

  friend bool operator==(const ArtifactByteRange&, const ArtifactByteRange&) = default;
};

enum class ArtifactDtype : uint8_t {
  kBool,
  kF4,
  kF6E2M3,
  kF6E3M2,
  kU8,
  kI8,
  kF8E5M2,
  kF8E4M3,
  kF8E8M0,
  kF8E4M3Fnuz,
  kF8E5M2Fnuz,
  kI16,
  kU16,
  kF16,
  kBf16,
  kI32,
  kU32,
  kF32,
  kC64,
  kF64,
  kI64,
  kU64,
};

using ArtifactShape = std::vector<uint64_t>;

struct ArtifactTensor {
  std::string name;
  ArtifactDtype dtype = ArtifactDtype::kF32;
  ArtifactShape shape;
  ArtifactByteRange data;
  uint64_t absolute_file_offset = 0;
  uint64_t packed_size_bytes = 0;
};

absl::StatusOr<ArtifactDtype> ParseArtifactDtype(std::string_view spelling);
std::string_view ArtifactDtypeName(ArtifactDtype dtype);
uint32_t ArtifactDtypeBits(ArtifactDtype dtype);
absl::StatusOr<uint64_t> PackedTensorBytes(ArtifactDtype dtype, const ArtifactShape& shape);

}  // namespace inferx::artifacts
