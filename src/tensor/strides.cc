#include "inferx/tensor/strides.h"

#include <absl/status/statusor.h>

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"

namespace inferx {

absl::StatusOr<Strides> Strides::CreateElements(std::span<const uint64_t> element_strides) {
  if (element_strides.size() > kMaxTensorRank) {
    return absl::InvalidArgumentError("strides.rank: rank exceeds 8");
  }
  Strides result;
  result.rank_ = static_cast<uint8_t>(element_strides.size());
  std::copy(element_strides.begin(), element_strides.end(), result.element_strides_.begin());
  return result;
}

absl::StatusOr<Strides> Strides::Contiguous(const Shape& shape) {
  Strides result;
  result.rank_ = shape.rank();
  if (std::find(shape.dimensions().begin(), shape.dimensions().end(), uint64_t{0}) !=
      shape.dimensions().end()) {
    // No element is reachable, so every empty shape is contiguous. A stable
    // all-one representation avoids overflowing products of irrelevant axes.
    std::fill_n(result.element_strides_.begin(), shape.rank(), uint64_t{1});
    return result;
  }
  uint64_t stride = 1;
  for (size_t axis = shape.rank(); axis > 0; --axis) {
    const size_t index = axis - 1;
    result.element_strides_[index] = stride;
    const uint64_t extent = shape.dim(index) == 0 ? 1 : shape.dim(index);
    absl::StatusOr<uint64_t> next = CheckedMul(stride, extent, "strides.contiguous");
    if (!next.ok()) {
      return next.status();
    }
    stride = *next;
  }
  return result;
}

absl::StatusOr<LayoutAnalysis> AnalyzeLayout(const Shape& shape, const Strides& strides,
                                             DType dtype) {
  if (shape.rank() != strides.rank()) {
    return absl::InvalidArgumentError("tensor.rank: shape and stride ranks differ");
  }
  absl::StatusOr<ByteCount> element_size = DTypeSize(dtype);
  if (!element_size.ok()) {
    return element_size.status();
  }
  absl::StatusOr<uint64_t> elements = shape.NumElements();
  if (!elements.ok()) {
    return elements.status();
  }

  if (*elements == 0) {
    for (size_t axis = 0; axis < shape.rank(); ++axis) {
      if (strides.elements(axis) == 0 && shape.dim(axis) > 1) {
        return absl::InvalidArgumentError(
            "strides.value: zero stride is allowed only for extent zero or one");
      }
    }
    LayoutAnalysis empty;
    empty.contiguous = true;
    empty.dense = true;
    empty.overlap = OverlapKind::kNonOverlapping;
    return empty;
  }

  LayoutAnalysis result;
  absl::StatusOr<Strides> expected = Strides::Contiguous(shape);
  if (!expected.ok()) {
    return expected.status();
  }
  result.contiguous = *elements == 0 || strides == *expected;

  std::array<std::pair<uint64_t, uint64_t>, kMaxTensorRank> active{};
  size_t active_count = 0;
  uint64_t maximum_offset = 0;
  for (size_t axis = 0; axis < shape.rank(); ++axis) {
    const uint64_t extent = shape.dim(axis);
    const uint64_t stride = strides.elements(axis);
    if (stride == 0 && extent > 1) {
      return absl::InvalidArgumentError(
          "strides.value: zero stride is allowed only for extent zero or one");
    }
    if (extent > 1) {
      active[active_count++] = {stride, extent};
      absl::StatusOr<uint64_t> contribution =
          CheckedMul(extent - 1, stride, "strides.maximum_offset");
      if (!contribution.ok()) {
        return contribution.status();
      }
      absl::StatusOr<uint64_t> next =
          CheckedAdd(maximum_offset, *contribution, "strides.maximum_offset");
      if (!next.ok()) {
        return next.status();
      }
      maximum_offset = *next;
    }
  }

  // The array is capped at rank 8. A direct insertion sort avoids heap work
  // and avoids libstdc++'s 16-element std::sort sentinel reading past this
  // smaller fixed array under aggressive inlining.
  for (size_t index = 1; index < active_count; ++index) {
    const auto value = active[index];
    size_t destination = index;
    while (destination > 0 && value < active[destination - 1]) {
      active[destination] = active[destination - 1];
      --destination;
    }
    active[destination] = value;
  }
  bool non_overlapping = true;
  uint64_t lower_span = 1;
  for (size_t index = 0; index < active_count; ++index) {
    const auto [stride, extent] = active[index];
    if (stride < lower_span) {
      non_overlapping = false;
      break;
    }
    absl::StatusOr<uint64_t> contribution = CheckedMul(extent - 1, stride, "strides.non_overlap");
    if (!contribution.ok()) {
      return contribution.status();
    }
    absl::StatusOr<uint64_t> next = CheckedAdd(*contribution, lower_span, "strides.non_overlap");
    if (!next.ok()) {
      return next.status();
    }
    lower_span = *next;
  }

  result.maximum_element_offset = maximum_offset;
  absl::StatusOr<uint64_t> maximum_bytes =
      CheckedMul(maximum_offset, element_size->value(), "strides.maximum_byte_offset");
  if (!maximum_bytes.ok()) {
    return maximum_bytes.status();
  }
  result.maximum_byte_offset = ByteCount(*maximum_bytes);
  absl::StatusOr<uint64_t> reachable =
      CheckedAdd(*maximum_bytes, element_size->value(), "strides.reachable_bytes");
  if (!reachable.ok()) {
    return reachable.status();
  }
  result.reachable_bytes = ByteCount(*reachable);
  result.overlap = non_overlapping ? OverlapKind::kNonOverlapping : OverlapKind::kMayOverlap;
  result.dense = non_overlapping && maximum_offset + 1 == *elements;
  return result;
}

}  // namespace inferx
