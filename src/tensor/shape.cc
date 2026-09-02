#include "inferx/tensor/shape.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"

namespace inferx {

absl::StatusOr<Shape> Shape::Create(std::span<const uint64_t> dimensions) {
  if (dimensions.size() > kMaxTensorRank) {
    return absl::InvalidArgumentError("shape.rank: rank exceeds 8");
  }
  Shape result;
  result.rank_ = static_cast<uint8_t>(dimensions.size());
  std::copy(dimensions.begin(), dimensions.end(), result.dimensions_.begin());
  return result;
}

absl::StatusOr<uint64_t> Shape::NumElements() const {
  for (uint64_t dimension : dimensions()) {
    if (dimension == 0) {
      return uint64_t{0};
    }
  }
  uint64_t product = 1;
  for (uint64_t dimension : dimensions()) {
    absl::StatusOr<uint64_t> next = CheckedMul(product, dimension, "shape.elements");
    if (!next.ok()) {
      return next.status();
    }
    product = *next;
  }
  return product;
}

absl::StatusOr<ByteCount> Shape::Bytes(Dtype dtype) const {
  absl::StatusOr<uint64_t> elements = NumElements();
  if (!elements.ok()) {
    return elements.status();
  }
  absl::StatusOr<ByteCount> element_size = DtypeSize(dtype);
  if (!element_size.ok()) {
    return element_size.status();
  }
  return CheckedByteSize(*elements, element_size->value(), "shape.bytes");
}

}  // namespace inferx
