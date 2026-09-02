#ifndef INFERX_SRC_OPS_VALIDATION_INTERNAL_H_
#define INFERX_SRC_OPS_VALIDATION_INTERNAL_H_

#include <cstdint>
#include <span>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::ops::internal {

struct TensorInterval {
  AllocationId allocation;
  uint64_t begin = 0;
  uint64_t size = 0;
};

[[nodiscard]] absl::Status ValidateTensor(const TensorView& tensor, Dtype dtype, uint8_t rank,
                                          std::string_view field, bool require_contiguous = true);
[[nodiscard]] absl::Status ValidateTensor(const MutableTensorView& tensor, Dtype dtype,
                                          uint8_t rank, std::string_view field,
                                          bool require_contiguous = true);
[[nodiscard]] absl::Status ValidateSameDevice(const TensorView& first, const TensorView& second,
                                              std::string_view field);
[[nodiscard]] absl::Status ValidateSameDevice(const TensorView& first,
                                              const MutableTensorView& second,
                                              std::string_view field);
[[nodiscard]] absl::StatusOr<TensorInterval> Interval(const TensorView& tensor);
[[nodiscard]] absl::StatusOr<TensorInterval> Interval(const MutableTensorView& tensor);
[[nodiscard]] bool Overlaps(const TensorInterval& left, const TensorInterval& right) noexcept;
[[nodiscard]] bool ExactAlias(const TensorInterval& left, const TensorInterval& right) noexcept;
[[nodiscard]] absl::Status ValidateDisjoint(const TensorView& input,
                                            const MutableTensorView& output,
                                            std::string_view field);

template <typename T>
[[nodiscard]] absl::StatusOr<std::span<const T>> HostSpan(const TensorView& tensor) {
  absl::StatusOr<std::span<const std::byte>> bytes = tensor.buffer().HostBytes();
  if (!bytes.ok()) return bytes.status();
  absl::StatusOr<uint64_t> elements = tensor.shape().NumElements();
  if (!elements.ok()) return elements.status();
  if (tensor.byte_offset().value() > bytes->size()) {
    return absl::InternalError("ops.tensor: validated byte offset is outside host buffer");
  }
  const auto* data = reinterpret_cast<const T*>(bytes->data() + tensor.byte_offset().value());
  return std::span<const T>(data, static_cast<size_t>(*elements));
}

template <typename T>
[[nodiscard]] absl::StatusOr<std::span<T>> HostSpan(const MutableTensorView& tensor) {
  absl::StatusOr<std::span<std::byte>> bytes = tensor.buffer().HostBytes();
  if (!bytes.ok()) return bytes.status();
  absl::StatusOr<uint64_t> elements = tensor.shape().NumElements();
  if (!elements.ok()) return elements.status();
  if (tensor.byte_offset().value() > bytes->size()) {
    return absl::InternalError("ops.tensor: validated byte offset is outside host buffer");
  }
  auto* data = reinterpret_cast<T*>(bytes->data() + tensor.byte_offset().value());
  return std::span<T>(data, static_cast<size_t>(*elements));
}

}  // namespace inferx::ops::internal

#endif  // INFERX_SRC_OPS_VALIDATION_INTERNAL_H_
