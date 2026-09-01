#include "cuda_tensor_checks.h"

#include <cstddef>

#include "absl/strings/str_cat.h"
#include "inferx/platform/cuda/cuda_buffer_access.h"

namespace inferx::kernels::cuda {

bool Overlaps(const TensorInterval& left, const TensorInterval& right) noexcept {
  if (left.allocation != right.allocation || left.size == 0 || right.size == 0) return false;
  if (left.begin <= right.begin) return right.begin - left.begin < left.size;
  return left.begin - right.begin < right.size;
}

bool ExactAlias(const TensorInterval& left, const TensorInterval& right) noexcept {
  return left.allocation == right.allocation && left.begin == right.begin &&
         left.size == right.size;
}

absl::Status ValidateBinaryAlias(const TensorView& left, const TensorView& right,
                                 const MutableTensorView& output, const char* field,
                                 bool allow_both_exact) {
  absl::StatusOr<TensorInterval> left_interval = MakeInterval(left);
  if (!left_interval.ok()) return left_interval.status();
  absl::StatusOr<TensorInterval> right_interval = MakeInterval(right);
  if (!right_interval.ok()) return right_interval.status();
  absl::StatusOr<TensorInterval> output_interval = MakeInterval(output);
  if (!output_interval.ok()) return output_interval.status();
  const bool exact_left = ExactAlias(*left_interval, *output_interval);
  const bool exact_right = ExactAlias(*right_interval, *output_interval);
  if ((Overlaps(*left_interval, *output_interval) && !exact_left) ||
      (Overlaps(*right_interval, *output_interval) && !exact_right) ||
      (!allow_both_exact && exact_left && Overlaps(*right_interval, *output_interval)) ||
      (!allow_both_exact && exact_right && Overlaps(*left_interval, *output_interval))) {
    return absl::InvalidArgumentError(absl::StrCat(field, ": unsupported tensor overlap"));
  }
  return absl::OkStatus();
}

const void* Address(const TensorView& tensor) noexcept {
  const auto* base = static_cast<const std::byte*>(::inferx::cuda::BufferAccess::Address(
      tensor.buffer()));
  return base == nullptr ? nullptr : base + tensor.byte_offset().value();
}

void* Address(const MutableTensorView& tensor) noexcept {
  auto* base =
      static_cast<std::byte*>(::inferx::cuda::BufferAccess::Address(tensor.buffer()));
  return base == nullptr ? nullptr : base + tensor.byte_offset().value();
}

StorageType ToKernelStorageType(DType dtype) noexcept {
  if (dtype == DType::kFloat16) return StorageType::kFloat16;
  if (dtype == DType::kBFloat16) return StorageType::kBFloat16;
  return StorageType::kFloat32;
}

bool SupportedStorage(DType dtype) noexcept {
  return dtype == DType::kFloat32 || dtype == DType::kFloat16 || dtype == DType::kBFloat16;
}

absl::Status ValidateIndptr(std::span<const int32_t> indptr, uint64_t expected_end) {
  if (indptr.empty() || indptr.front() != 0 || indptr.back() < 0 ||
      static_cast<uint64_t>(indptr.back()) != expected_end) {
    return absl::InvalidArgumentError("kernels_cuda.indptr: invalid endpoints");
  }
  for (size_t index = 1; index < indptr.size(); ++index) {
    if (indptr[index] < indptr[index - 1]) {
      return absl::InvalidArgumentError("kernels_cuda.indptr: offsets must be nondecreasing");
    }
  }
  return absl::OkStatus();
}

namespace {

// Mirrors the M2 sticky-fault set (platform/cuda IsCudaStickyFault): errors
// that poison the device context rather than failing one launch.
bool IsStickyLaunchFault(cudaError_t error) noexcept {
  switch (error) {
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
    case cudaErrorLaunchTimeout:
    case cudaErrorAssert:
    case cudaErrorContextIsDestroyed:
      return true;
    default:
      return false;
  }
}

}  // namespace

absl::Status CheckLaunchResult(cudaError_t error, const char* operation,
                               const CudaLaunchContext& context) {
  if (error == cudaSuccess) return absl::OkStatus();
  absl::StatusCode code = absl::StatusCode::kInternal;
  switch (error) {
    case cudaErrorInvalidDevice:
    case cudaErrorInvalidValue:
    case cudaErrorInvalidConfiguration:
      code = absl::StatusCode::kInvalidArgument;
      break;
    case cudaErrorMemoryAllocation:
    case cudaErrorLaunchOutOfResources:
      code = absl::StatusCode::kResourceExhausted;
      break;
    default:
      break;
  }
  if (context.observer != nullptr &&
      (IsStickyLaunchFault(error) || error == cudaErrorInvalidResourceHandle)) {
    context.observer->OnKernelFailure(operation, /*sticky=*/true);
  }
  return absl::Status(
      code, absl::StrCat(operation, ": cuda error ", static_cast<int>(error), " (",
                         cudaGetErrorString(error), ")"));
}

}  // namespace inferx::kernels::cuda
