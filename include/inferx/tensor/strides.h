// Element-unit tensor strides and conservative layout analysis.
#ifndef INFERX_TENSOR_STRIDES_H_
#define INFERX_TENSOR_STRIDES_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/statusor.h"
#include "inferx/base/token.h"
#include "inferx/tensor/dtype.h"
#include "inferx/tensor/shape.h"

namespace inferx {

class Strides {
 public:
  [[nodiscard]] static absl::StatusOr<Strides> CreateElements(
      std::span<const uint64_t> element_strides);
  [[nodiscard]] static absl::StatusOr<Strides> Contiguous(const Shape& shape);

  [[nodiscard]] constexpr uint8_t rank() const noexcept { return rank_; }
  [[nodiscard]] constexpr uint64_t elements(size_t axis) const noexcept {
    return element_strides_[axis];
  }
  [[nodiscard]] constexpr std::span<const uint64_t> values() const noexcept {
    return std::span<const uint64_t>(element_strides_.data(), rank_);
  }

  friend constexpr bool operator==(const Strides&, const Strides&) = default;

 private:
  std::array<uint64_t, kMaxTensorRank> element_strides_{};
  uint8_t rank_ = 0;
};

enum class OverlapKind : uint8_t { kNonOverlapping = 0, kMayOverlap = 1 };

struct LayoutAnalysis {
  bool contiguous = false;
  bool dense = false;
  OverlapKind overlap = OverlapKind::kMayOverlap;
  uint64_t maximum_element_offset = 0;
  ByteCount maximum_byte_offset = ByteCount(0);
  ByteCount reachable_bytes = ByteCount(0);
};

[[nodiscard]] absl::StatusOr<LayoutAnalysis> AnalyzeLayout(const Shape& shape,
                                                           const Strides& strides, Dtype dtype);

}  // namespace inferx

#endif  // INFERX_TENSOR_STRIDES_H_
