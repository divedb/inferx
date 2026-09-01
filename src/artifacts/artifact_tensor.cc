#include "inferx/artifacts/artifact_tensor.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string_view>

#include "absl/status/status.h"

namespace inferx::artifacts {
namespace {

struct DTypeInfo {
  std::string_view name;
  ArtifactDType dtype;
  uint32_t bits;
};

constexpr std::array kDTypes = {
    DTypeInfo{"BOOL", ArtifactDType::kBool, 8},
    DTypeInfo{"F4", ArtifactDType::kF4, 4},
    DTypeInfo{"F6_E2M3", ArtifactDType::kF6E2M3, 6},
    DTypeInfo{"F6_E3M2", ArtifactDType::kF6E3M2, 6},
    DTypeInfo{"U8", ArtifactDType::kU8, 8},
    DTypeInfo{"I8", ArtifactDType::kI8, 8},
    DTypeInfo{"F8_E5M2", ArtifactDType::kF8E5M2, 8},
    DTypeInfo{"F8_E4M3", ArtifactDType::kF8E4M3, 8},
    DTypeInfo{"F8_E8M0", ArtifactDType::kF8E8M0, 8},
    DTypeInfo{"F8_E4M3FNUZ", ArtifactDType::kF8E4M3Fnuz, 8},
    DTypeInfo{"F8_E5M2FNUZ", ArtifactDType::kF8E5M2Fnuz, 8},
    DTypeInfo{"I16", ArtifactDType::kI16, 16},
    DTypeInfo{"U16", ArtifactDType::kU16, 16},
    DTypeInfo{"F16", ArtifactDType::kF16, 16},
    DTypeInfo{"BF16", ArtifactDType::kBf16, 16},
    DTypeInfo{"I32", ArtifactDType::kI32, 32},
    DTypeInfo{"U32", ArtifactDType::kU32, 32},
    DTypeInfo{"F32", ArtifactDType::kF32, 32},
    DTypeInfo{"C64", ArtifactDType::kC64, 64},
    DTypeInfo{"F64", ArtifactDType::kF64, 64},
    DTypeInfo{"I64", ArtifactDType::kI64, 64},
    DTypeInfo{"U64", ArtifactDType::kU64, 64},
};

}  // namespace

absl::StatusOr<ArtifactDType> ParseArtifactDType(std::string_view spelling) {
  for (const auto& info : kDTypes) {
    if (info.name == spelling) return info.dtype;
  }
  return absl::InvalidArgumentError("unknown safetensors dtype");
}

std::string_view ArtifactDTypeName(ArtifactDType dtype) {
  for (const auto& info : kDTypes) {
    if (info.dtype == dtype) return info.name;
  }
  return "UNKNOWN";
}

uint32_t ArtifactDTypeBits(ArtifactDType dtype) {
  for (const auto& info : kDTypes) {
    if (info.dtype == dtype) return info.bits;
  }
  return 0;
}

absl::StatusOr<uint64_t> PackedTensorBytes(ArtifactDType dtype, const ArtifactShape& shape) {
  uint64_t elements = 1;
  for (const uint64_t dimension : shape) {
    if (dimension == 0) return uint64_t{0};
    if (elements > std::numeric_limits<uint64_t>::max() / dimension) {
      return absl::OutOfRangeError("tensor element count overflows uint64");
    }
    elements *= dimension;
  }
  const uint64_t bits = ArtifactDTypeBits(dtype);
  if (bits == 0 || elements > (std::numeric_limits<uint64_t>::max() - 7) / bits) {
    return absl::OutOfRangeError("packed tensor byte size overflows uint64");
  }
  return (elements * bits + 7) / 8;
}

}  // namespace inferx::artifacts
