// Checked immutable and mutable tensor views.
#ifndef INFERX_TENSOR_TENSOR_VIEW_H_
#define INFERX_TENSOR_TENSOR_VIEW_H_

#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/token.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/dtype.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"

namespace inferx {

class TensorView {
 public:
  // Inert empty view; created views come from Create/Slice/Permute/Reshape.
  TensorView() noexcept = default;
  [[nodiscard]] static absl::StatusOr<TensorView> Create(BufferView buffer, Dtype dtype,
                                                         Shape shape, Strides strides,
                                                         ByteCount byte_offset = ByteCount(0));

  [[nodiscard]] const BufferView& buffer() const noexcept { return buffer_; }
  [[nodiscard]] Dtype dtype() const noexcept { return dtype_; }
  [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
  [[nodiscard]] const Strides& strides() const noexcept { return strides_; }
  [[nodiscard]] ByteCount byte_offset() const noexcept { return byte_offset_; }
  [[nodiscard]] const LayoutAnalysis& layout() const noexcept { return layout_; }

  [[nodiscard]] absl::StatusOr<TensorView> Slice(size_t axis, uint64_t begin, uint64_t end,
                                                 uint64_t step = 1) const;
  [[nodiscard]] absl::StatusOr<TensorView> Permute(std::span<const uint8_t> axes) const;
  [[nodiscard]] absl::StatusOr<TensorView> Reshape(const Shape& shape) const;

 private:
  friend class MutableTensorView;
  TensorView(BufferView buffer, Dtype dtype, Shape shape, Strides strides, ByteCount byte_offset,
             LayoutAnalysis layout) noexcept;

  BufferView buffer_;
  Dtype dtype_ = Dtype::kFloat32;
  Shape shape_;
  Strides strides_;
  ByteCount byte_offset_ = ByteCount(0);
  LayoutAnalysis layout_;
};

class MutableTensorView {
 public:
  // Inert empty view; created views come from Create.
  MutableTensorView() noexcept = default;
  [[nodiscard]] static absl::StatusOr<MutableTensorView> Create(
      MutableBufferView buffer, Dtype dtype, Shape shape, Strides strides,
      ByteCount byte_offset = ByteCount(0));

  [[nodiscard]] const MutableBufferView& buffer() const noexcept { return buffer_; }
  [[nodiscard]] Dtype dtype() const noexcept { return dtype_; }
  [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
  [[nodiscard]] const Strides& strides() const noexcept { return strides_; }
  [[nodiscard]] ByteCount byte_offset() const noexcept { return byte_offset_; }
  [[nodiscard]] const LayoutAnalysis& layout() const noexcept { return layout_; }
  [[nodiscard]] TensorView AsConst() const;

 private:
  MutableTensorView(MutableBufferView buffer, Dtype dtype, Shape shape, Strides strides,
                    ByteCount byte_offset, LayoutAnalysis layout) noexcept;

  MutableBufferView buffer_;
  Dtype dtype_ = Dtype::kFloat32;
  Shape shape_;
  Strides strides_;
  ByteCount byte_offset_ = ByteCount(0);
  LayoutAnalysis layout_;
};

[[nodiscard]] absl::Status CopyTensorCpu(const TensorView& source,
                                         const MutableTensorView& destination);
[[nodiscard]] absl::Status FillBufferCpu(MutableBufferView destination, uint8_t value);

}  // namespace inferx

#endif  // INFERX_TENSOR_TENSOR_VIEW_H_
