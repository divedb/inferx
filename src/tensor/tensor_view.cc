#include "inferx/tensor/tensor_view.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"

namespace inferx {
namespace {

absl::Status ValidateViewBounds(const BufferView& buffer, DType dtype, const Shape& shape,
                                const Strides& strides, ByteCount byte_offset,
                                LayoutAnalysis* analysis) {
  absl::StatusOr<ByteCount> dtype_size = DTypeSize(dtype);
  if (!dtype_size.ok()) {
    return dtype_size.status();
  }
  if (byte_offset.value() % dtype_size->value() != 0) {
    return absl::InvalidArgumentError("tensor.byte_offset: offset is not aligned to dtype width");
  }
  if (buffer.alignment().value() < dtype_size->value() && buffer.size().value() != 0) {
    return absl::InvalidArgumentError("tensor.buffer: buffer alignment is insufficient for dtype");
  }
  absl::StatusOr<LayoutAnalysis> layout = AnalyzeLayout(shape, strides, dtype);
  if (!layout.ok()) {
    return layout.status();
  }
  absl::Status status =
      ByteRange{byte_offset, layout->reachable_bytes}.ValidateWithin(buffer.size());
  if (!status.ok()) {
    return status;
  }
  *analysis = *layout;
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> LogicalElementOffset(const Shape& shape, const Strides& strides,
                                              uint64_t linear) {
  uint64_t result = 0;
  for (size_t reverse_axis = shape.rank(); reverse_axis > 0; --reverse_axis) {
    const size_t axis = reverse_axis - 1;
    const uint64_t extent = shape.dim(axis);
    const uint64_t index = extent == 0 ? 0 : linear % extent;
    if (extent != 0) {
      linear /= extent;
    }
    absl::StatusOr<uint64_t> term = CheckedMul(index, strides.elements(axis), "tensor.copy.offset");
    if (!term.ok()) {
      return term.status();
    }
    absl::StatusOr<uint64_t> next = CheckedAdd(result, *term, "tensor.copy.offset");
    if (!next.ok()) {
      return next.status();
    }
    result = *next;
  }
  return result;
}

bool RangesOverlap(uint64_t first_begin, uint64_t first_size, uint64_t second_begin,
                   uint64_t second_size) {
  if (first_size == 0 || second_size == 0) {
    return false;
  }
  return first_begin < second_begin + second_size && second_begin < first_begin + first_size;
}

}  // namespace

TensorView::TensorView(BufferView buffer, DType dtype, Shape shape, Strides strides,
                       ByteCount byte_offset, LayoutAnalysis layout) noexcept
    : buffer_(buffer),
      dtype_(dtype),
      shape_(shape),
      strides_(strides),
      byte_offset_(byte_offset),
      layout_(layout) {}

absl::StatusOr<TensorView> TensorView::Create(BufferView buffer, DType dtype, Shape shape,
                                              Strides strides, ByteCount byte_offset) {
  LayoutAnalysis analysis;
  absl::Status status = ValidateViewBounds(buffer, dtype, shape, strides, byte_offset, &analysis);
  if (!status.ok()) {
    return status;
  }
  return TensorView(buffer, dtype, shape, strides, byte_offset, analysis);
}

absl::StatusOr<TensorView> TensorView::Slice(size_t axis, uint64_t begin, uint64_t end,
                                             uint64_t step) const {
  if (axis >= shape_.rank()) {
    return absl::InvalidArgumentError("tensor.slice.axis: axis is out of range");
  }
  if (step == 0) {
    return absl::InvalidArgumentError("tensor.slice.step: step must be positive");
  }
  if (begin > end || end > shape_.dim(axis)) {
    return absl::InvalidArgumentError("tensor.slice.range: expected begin <= end <= dimension");
  }
  const uint64_t distance = end - begin;
  const uint64_t new_extent = distance == 0 ? 0 : 1 + (distance - 1) / step;

  std::array<uint64_t, kMaxTensorRank> dimensions{};
  std::array<uint64_t, kMaxTensorRank> stride_values{};
  std::copy(shape_.dimensions().begin(), shape_.dimensions().end(), dimensions.begin());
  std::copy(strides_.values().begin(), strides_.values().end(), stride_values.begin());
  dimensions[axis] = new_extent;
  absl::StatusOr<uint64_t> new_stride =
      CheckedMul(stride_values[axis], step, "tensor.slice.stride");
  if (!new_stride.ok()) {
    return new_stride.status();
  }
  stride_values[axis] = *new_stride;

  absl::StatusOr<ByteCount> element_size = DTypeSize(dtype_);
  if (!element_size.ok()) {
    return element_size.status();
  }
  absl::StatusOr<uint64_t> element_delta =
      CheckedMul(begin, strides_.elements(axis), "tensor.slice.offset");
  if (!element_delta.ok()) {
    return element_delta.status();
  }
  absl::StatusOr<uint64_t> byte_delta =
      CheckedMul(*element_delta, element_size->value(), "tensor.slice.offset");
  if (!byte_delta.ok()) {
    return byte_delta.status();
  }
  absl::StatusOr<uint64_t> offset =
      CheckedAdd(byte_offset_.value(), *byte_delta, "tensor.slice.offset");
  if (!offset.ok()) {
    return offset.status();
  }
  absl::StatusOr<Shape> shape =
      Shape::Create(std::span<const uint64_t>(dimensions.data(), shape_.rank()));
  if (!shape.ok()) {
    return shape.status();
  }
  absl::StatusOr<Strides> strides =
      Strides::CreateElements(std::span<const uint64_t>(stride_values.data(), shape_.rank()));
  if (!strides.ok()) {
    return strides.status();
  }
  return Create(buffer_, dtype_, *shape, *strides, ByteCount(*offset));
}

absl::StatusOr<TensorView> TensorView::Permute(std::span<const uint8_t> axes) const {
  if (axes.size() != shape_.rank()) {
    return absl::InvalidArgumentError(
        "tensor.permute.axes: permutation rank differs from tensor rank");
  }
  std::array<bool, kMaxTensorRank> seen{};
  std::array<uint64_t, kMaxTensorRank> dimensions{};
  std::array<uint64_t, kMaxTensorRank> stride_values{};
  for (size_t index = 0; index < axes.size(); ++index) {
    const size_t axis = axes[index];
    if (axis >= shape_.rank() || seen[axis]) {
      return absl::InvalidArgumentError(
          "tensor.permute.axes: axes must be a complete unique permutation");
    }
    seen[axis] = true;
    dimensions[index] = shape_.dim(axis);
    stride_values[index] = strides_.elements(axis);
  }
  absl::StatusOr<Shape> shape =
      Shape::Create(std::span<const uint64_t>(dimensions.data(), shape_.rank()));
  if (!shape.ok()) {
    return shape.status();
  }
  absl::StatusOr<Strides> strides =
      Strides::CreateElements(std::span<const uint64_t>(stride_values.data(), shape_.rank()));
  if (!strides.ok()) {
    return strides.status();
  }
  return Create(buffer_, dtype_, *shape, *strides, byte_offset_);
}

absl::StatusOr<TensorView> TensorView::Reshape(const Shape& shape) const {
  if (!layout_.contiguous) {
    return absl::FailedPreconditionError("tensor.reshape: input must be contiguous");
  }
  absl::StatusOr<uint64_t> old_elements = shape_.NumElements();
  if (!old_elements.ok()) {
    return old_elements.status();
  }
  absl::StatusOr<uint64_t> new_elements = shape.NumElements();
  if (!new_elements.ok()) {
    return new_elements.status();
  }
  if (*old_elements != *new_elements) {
    return absl::InvalidArgumentError("tensor.reshape: element count must match exactly");
  }
  absl::StatusOr<Strides> strides = Strides::Contiguous(shape);
  if (!strides.ok()) {
    return strides.status();
  }
  return Create(buffer_, dtype_, shape, *strides, byte_offset_);
}

MutableTensorView::MutableTensorView(MutableBufferView buffer, DType dtype, Shape shape,
                                     Strides strides, ByteCount byte_offset,
                                     LayoutAnalysis layout) noexcept
    : buffer_(buffer),
      dtype_(dtype),
      shape_(shape),
      strides_(strides),
      byte_offset_(byte_offset),
      layout_(layout) {}

absl::StatusOr<MutableTensorView> MutableTensorView::Create(MutableBufferView buffer, DType dtype,
                                                            Shape shape, Strides strides,
                                                            ByteCount byte_offset) {
  LayoutAnalysis analysis;
  absl::Status status =
      ValidateViewBounds(buffer.AsConst(), dtype, shape, strides, byte_offset, &analysis);
  if (!status.ok()) {
    return status;
  }
  if (analysis.overlap != OverlapKind::kNonOverlapping) {
    return absl::InvalidArgumentError("tensor.mutable: overlapping layouts cannot be mutable");
  }
  return MutableTensorView(buffer, dtype, shape, strides, byte_offset, analysis);
}

TensorView MutableTensorView::AsConst() const {
  return TensorView(buffer_.AsConst(), dtype_, shape_, strides_, byte_offset_, layout_);
}

absl::Status FillBufferCpu(MutableBufferView destination, uint8_t value) {
  absl::StatusOr<std::span<std::byte>> bytes = destination.HostBytes();
  if (!bytes.ok()) {
    return bytes.status();
  }
  std::memset(bytes->data(), value, bytes->size());
  return absl::OkStatus();
}

absl::Status CopyTensorCpu(const TensorView& source, const MutableTensorView& destination) {
  if (source.dtype() != destination.dtype()) {
    return absl::InvalidArgumentError("tensor.copy.dtype: dtypes must match");
  }
  if (source.shape() != destination.shape()) {
    return absl::InvalidArgumentError("tensor.copy.shape: shapes must match");
  }
  absl::StatusOr<std::span<const std::byte>> source_bytes = source.buffer().HostBytes();
  if (!source_bytes.ok()) {
    return source_bytes.status();
  }
  absl::StatusOr<std::span<std::byte>> destination_bytes = destination.buffer().HostBytes();
  if (!destination_bytes.ok()) {
    return destination_bytes.status();
  }
  absl::StatusOr<ByteCount> element_size = DTypeSize(source.dtype());
  if (!element_size.ok()) {
    return element_size.status();
  }
  absl::StatusOr<uint64_t> element_count = source.shape().NumElements();
  if (!element_count.ok()) {
    return element_count.status();
  }
  if (*element_count == 0) {
    return absl::OkStatus();
  }

  if (source.layout().contiguous && destination.layout().contiguous) {
    absl::StatusOr<ByteCount> size = source.shape().Bytes(source.dtype());
    if (!size.ok()) {
      return size.status();
    }
    absl::StatusOr<size_t> copy_size = CheckedNarrow<size_t>(size->value(), "tensor.copy.bytes");
    if (!copy_size.ok()) {
      return copy_size.status();
    }
    std::memmove(destination_bytes->data() + destination.byte_offset().value(),
                 source_bytes->data() + source.byte_offset().value(), *copy_size);
    return absl::OkStatus();
  }

  if (source.buffer().allocation_id() == destination.buffer().allocation_id()) {
    const uint64_t source_begin =
        source.buffer().range().offset.value() + source.byte_offset().value();
    const uint64_t destination_begin =
        destination.buffer().range().offset.value() + destination.byte_offset().value();
    if (RangesOverlap(source_begin, source.layout().reachable_bytes.value(), destination_begin,
                      destination.layout().reachable_bytes.value())) {
      return absl::InvalidArgumentError(
          "tensor.copy.alias: overlapping strided copies are unsupported");
    }
  }

  absl::StatusOr<size_t> width =
      CheckedNarrow<size_t>(element_size->value(), "tensor.copy.element_size");
  if (!width.ok()) {
    return width.status();
  }
  for (uint64_t linear = 0; linear < *element_count; ++linear) {
    absl::StatusOr<uint64_t> source_element =
        LogicalElementOffset(source.shape(), source.strides(), linear);
    if (!source_element.ok()) {
      return source_element.status();
    }
    absl::StatusOr<uint64_t> destination_element =
        LogicalElementOffset(destination.shape(), destination.strides(), linear);
    if (!destination_element.ok()) {
      return destination_element.status();
    }
    const uint64_t source_offset =
        source.byte_offset().value() + *source_element * element_size->value();
    const uint64_t destination_offset =
        destination.byte_offset().value() + *destination_element * element_size->value();
    std::memcpy(destination_bytes->data() + destination_offset,
                source_bytes->data() + source_offset, *width);
  }
  return absl::OkStatus();
}

}  // namespace inferx
