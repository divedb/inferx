// Fixed-rank checked tensor shape.
#ifndef INFERX_TENSOR_SHAPE_H_
#define INFERX_TENSOR_SHAPE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/statusor.h"
#include "inferx/base/token.h"
#include "inferx/tensor/dtype.h"

namespace inferx {

inline constexpr size_t kMaxTensorRank = 8;

class Shape {
 public:
  [[nodiscard]] static absl::StatusOr<Shape> Create(std::span<const uint64_t> dimensions);

  [[nodiscard]] constexpr uint8_t rank() const noexcept { return rank_; }
  [[nodiscard]] constexpr uint64_t dim(size_t axis) const noexcept { return dimensions_[axis]; }
  [[nodiscard]] constexpr std::span<const uint64_t> dimensions() const noexcept {
    return std::span<const uint64_t>(dimensions_.data(), rank_);
  }
  [[nodiscard]] absl::StatusOr<uint64_t> NumElements() const;
  [[nodiscard]] absl::StatusOr<ByteCount> Bytes(DType dtype) const;

  friend constexpr bool operator==(const Shape&, const Shape&) = default;

 private:
  std::array<uint64_t, kMaxTensorRank> dimensions_{};
  uint8_t rank_ = 0;
};

}  // namespace inferx

#endif  // INFERX_TENSOR_SHAPE_H_
