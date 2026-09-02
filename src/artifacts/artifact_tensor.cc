#include "inferx/artifacts/artifact_tensor.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string_view>

#include "absl/status/status.h"

namespace inferx::artifacts {
namespace {

struct DtypeInfo {
  std::string_view name;
  ArtifactDtype dtype;
  uint32_t bits;
};

constexpr std::array kDtypes = {
    DtypeInfo{"BOOL", ArtifactDtype::kBool, 8},
    DtypeInfo{"F4", ArtifactDtype::kF4, 4},
    DtypeInfo{"F6_E2M3", ArtifactDtype::kF6E2M3, 6},
    DtypeInfo{"F6_E3M2", ArtifactDtype::kF6E3M2, 6},
    DtypeInfo{"U8", ArtifactDtype::kU8, 8},
    DtypeInfo{"I8", ArtifactDtype::kI8, 8},
    DtypeInfo{"F8_E5M2", ArtifactDtype::kF8E5M2, 8},
    DtypeInfo{"F8_E4M3", ArtifactDtype::kF8E4M3, 8},
    DtypeInfo{"F8_E8M0", ArtifactDtype::kF8E8M0, 8},
    DtypeInfo{"F8_E4M3FNUZ", ArtifactDtype::kF8E4M3Fnuz, 8},
    DtypeInfo{"F8_E5M2FNUZ", ArtifactDtype::kF8E5M2Fnuz, 8},
    DtypeInfo{"I16", ArtifactDtype::kI16, 16},
    DtypeInfo{"U16", ArtifactDtype::kU16, 16},
    DtypeInfo{"F16", ArtifactDtype::kF16, 16},
    DtypeInfo{"BF16", ArtifactDtype::kBf16, 16},
    DtypeInfo{"I32", ArtifactDtype::kI32, 32},
    DtypeInfo{"U32", ArtifactDtype::kU32, 32},
    DtypeInfo{"F32", ArtifactDtype::kF32, 32},
    DtypeInfo{"C64", ArtifactDtype::kC64, 64},
    DtypeInfo{"F64", ArtifactDtype::kF64, 64},
    DtypeInfo{"I64", ArtifactDtype::kI64, 64},
    DtypeInfo{"U64", ArtifactDtype::kU64, 64},
};

}  // namespace

absl::StatusOr<ArtifactDtype> ParseArtifactDtype(std::string_view spelling) {
  for (const auto& info : kDtypes) {
    if (info.name == spelling) return info.dtype;
  }
  return absl::InvalidArgumentError("unknown safetensors dtype");
}

std::string_view ArtifactDtypeName(ArtifactDtype dtype) {
  for (const auto& info : kDtypes) {
    if (info.dtype == dtype) return info.name;
  }
  return "UNKNOWN";
}

uint32_t ArtifactDtypeBits(ArtifactDtype dtype) {
  for (const auto& info : kDtypes) {
    if (info.dtype == dtype) return info.bits;
  }
  return 0;
}

absl::StatusOr<uint64_t> PackedTensorBytes(ArtifactDtype dtype, const ArtifactShape& shape) {
  uint64_t elements = 1;
  for (const uint64_t dimension : shape) {
    if (dimension == 0) return uint64_t{0};
    if (elements > std::numeric_limits<uint64_t>::max() / dimension) {
      return absl::OutOfRangeError("tensor element count overflows uint64");
    }
    elements *= dimension;
  }
  const uint64_t bits = ArtifactDtypeBits(dtype);
  if (bits == 0 || elements > (std::numeric_limits<uint64_t>::max() - 7) / bits) {
    return absl::OutOfRangeError("packed tensor byte size overflows uint64");
  }
  return (elements * bits + 7) / 8;
}

}  // namespace inferx::artifacts
