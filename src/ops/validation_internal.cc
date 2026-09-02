#include "src/ops/validation_internal.h"

#include <cstdint>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"

namespace inferx::ops::internal {
namespace {

template <typename View>
absl::Status ValidateTensorImpl(const View& tensor, Dtype dtype, uint8_t rank,
                                std::string_view field, bool require_contiguous) {
  if (tensor.dtype() != dtype) {
    return absl::InvalidArgumentError(absl::StrCat(field, ": unexpected dtype"));
  }
  if (tensor.shape().rank() != rank) {
    return absl::InvalidArgumentError(absl::StrCat(field, ": unexpected rank"));
  }
  if (require_contiguous && !tensor.layout().contiguous) {
    return absl::UnimplementedError(absl::StrCat(field, ": contiguous layout is required"));
  }
  return absl::OkStatus();
}

template <typename View>
absl::StatusOr<TensorInterval> IntervalImpl(const View& tensor) {
  absl::StatusOr<uint64_t> begin =
      CheckedAdd(tensor.buffer().range().offset.value(), tensor.byte_offset().value(),
                 "ops.tensor_interval.begin");
  if (!begin.ok()) return begin.status();
  return TensorInterval{tensor.buffer().allocation_id(), *begin,
                        tensor.layout().reachable_bytes.value()};
}

}  // namespace

absl::Status ValidateTensor(const TensorView& tensor, Dtype dtype, uint8_t rank,
                            std::string_view field, bool require_contiguous) {
  return ValidateTensorImpl(tensor, dtype, rank, field, require_contiguous);
}

absl::Status ValidateTensor(const MutableTensorView& tensor, Dtype dtype, uint8_t rank,
                            std::string_view field, bool require_contiguous) {
  return ValidateTensorImpl(tensor, dtype, rank, field, require_contiguous);
}

absl::Status ValidateSameDevice(const TensorView& first, const TensorView& second,
                                std::string_view field) {
  if (first.buffer().device() != second.buffer().device()) {
    return absl::InvalidArgumentError(absl::StrCat(field, ": tensors are on different devices"));
  }
  return absl::OkStatus();
}

absl::Status ValidateSameDevice(const TensorView& first, const MutableTensorView& second,
                                std::string_view field) {
  if (first.buffer().device() != second.buffer().device()) {
    return absl::InvalidArgumentError(absl::StrCat(field, ": tensors are on different devices"));
  }
  return absl::OkStatus();
}

absl::StatusOr<TensorInterval> Interval(const TensorView& tensor) { return IntervalImpl(tensor); }

absl::StatusOr<TensorInterval> Interval(const MutableTensorView& tensor) {
  return IntervalImpl(tensor);
}

bool Overlaps(const TensorInterval& left, const TensorInterval& right) noexcept {
  if (left.allocation != right.allocation || left.size == 0 || right.size == 0) return false;
  if (left.begin <= right.begin) return right.begin - left.begin < left.size;
  return left.begin - right.begin < right.size;
}

bool ExactAlias(const TensorInterval& left, const TensorInterval& right) noexcept {
  return left.allocation == right.allocation && left.begin == right.begin &&
         left.size == right.size;
}

absl::Status ValidateDisjoint(const TensorView& input, const MutableTensorView& output,
                              std::string_view field) {
  absl::StatusOr<TensorInterval> input_interval = Interval(input);
  if (!input_interval.ok()) return input_interval.status();
  absl::StatusOr<TensorInterval> output_interval = Interval(output);
  if (!output_interval.ok()) return output_interval.status();
  if (Overlaps(*input_interval, *output_interval)) {
    return absl::InvalidArgumentError(absl::StrCat(field, ": tensor ranges overlap"));
  }
  return absl::OkStatus();
}

}  // namespace inferx::ops::internal
