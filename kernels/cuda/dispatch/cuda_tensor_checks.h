// Shared request-validation and launch-result helpers for kernels/cuda
// providers. This is the single copy of the alias/overlap/device checks that
// the dispatch glue centralized; the legacy copies in platform/cuda stay
// untouched during the architecture transition (ADR 0031).
#ifndef INFERX_KERNELS_CUDA_DISPATCH_CUDA_TENSOR_CHECKS_H_
#define INFERX_KERNELS_CUDA_DISPATCH_CUDA_TENSOR_CHECKS_H_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cuda_launch_context.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/id.h"
#include "inferx/kernels/cuda/storage_type.h"
#include "inferx/tensor/dtype.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::kernels::cuda {

struct TensorInterval {
  AllocationId allocation;
  uint64_t begin = 0;
  uint64_t size = 0;
};

template <typename View>
absl::StatusOr<TensorInterval> MakeInterval(const View& tensor) {
  absl::StatusOr<uint64_t> begin =
      CheckedAdd(tensor.buffer().range().offset.value(), tensor.byte_offset().value(),
                 "kernels_cuda.tensor_interval.begin");
  if (!begin.ok()) return begin.status();
  return TensorInterval{tensor.buffer().allocation_id(), *begin,
                        tensor.layout().reachable_bytes.value()};
}

[[nodiscard]] bool Overlaps(const TensorInterval& left, const TensorInterval& right) noexcept;
[[nodiscard]] bool ExactAlias(const TensorInterval& left, const TensorInterval& right) noexcept;

template <typename First, typename Second>
absl::Status RejectOverlap(const First& first, const Second& second, const char* field) {
  absl::StatusOr<TensorInterval> first_interval = MakeInterval(first);
  if (!first_interval.ok()) return first_interval.status();
  absl::StatusOr<TensorInterval> second_interval = MakeInterval(second);
  if (!second_interval.ok()) return second_interval.status();
  if (Overlaps(*first_interval, *second_interval)) {
    return absl::InvalidArgumentError(absl::StrCat(field, ": tensor ranges overlap"));
  }
  return absl::OkStatus();
}

template <typename Input>
absl::Status RequireExactOrDisjoint(const Input& input, const MutableTensorView& output,
                                    const char* field) {
  absl::StatusOr<TensorInterval> input_interval = MakeInterval(input);
  if (!input_interval.ok()) return input_interval.status();
  absl::StatusOr<TensorInterval> output_interval = MakeInterval(output);
  if (!output_interval.ok()) return output_interval.status();
  if (Overlaps(*input_interval, *output_interval) &&
      !ExactAlias(*input_interval, *output_interval)) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, ": only an exact in-place alias is allowed"));
  }
  return absl::OkStatus();
}

absl::Status ValidateBinaryAlias(const TensorView& left, const TensorView& right,
                                 const MutableTensorView& output, const char* field,
                                 bool allow_both_exact);

[[nodiscard]] const void* Address(const TensorView& tensor) noexcept;
[[nodiscard]] void* Address(const MutableTensorView& tensor) noexcept;

[[nodiscard]] StorageType ToKernelStorageType(Dtype dtype) noexcept;
[[nodiscard]] bool SupportedStorage(Dtype dtype) noexcept;

template <typename View>
absl::Status ValidateCudaTensor(const View& tensor, uint8_t rank, const char* field) {
  if (!SupportedStorage(tensor.dtype())) {
    return absl::UnimplementedError(std::string(field) + ": FP32, FP16, or BF16 is required");
  }
  if (tensor.shape().rank() != rank || !tensor.layout().contiguous) {
    return absl::UnimplementedError(std::string(field) + ": contiguous rank mismatch");
  }
  if (tensor.buffer().device().kind != DeviceKind::kCuda ||
      tensor.buffer().memory_kind() != MemoryKind::kDevice) {
    return absl::InvalidArgumentError(std::string(field) + ": CUDA device memory is required");
  }
  return absl::OkStatus();
}

template <typename View>
absl::Status ValidateDeviceMatch(const View& tensor, DeviceId device) {
  if (tensor.buffer().device().ordinal != device) {
    return absl::InvalidArgumentError("kernels_cuda.device: tensor and context differ");
  }
  return absl::OkStatus();
}

absl::Status ValidateIndptr(std::span<const int32_t> indptr, uint64_t expected_end);

// Wraps a raw kernel-launch result: maps the CUDA error to an absl status,
// forwards sticky faults to the context observer (when present), and reports
// success as "queued" semantics — completion is the caller's fence concern.
absl::Status CheckLaunchResult(cudaError_t error, const char* operation,
                               const CudaLaunchContext& context);

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_DISPATCH_CUDA_TENSOR_CHECKS_H_
