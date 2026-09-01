#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/ops/gemm.h"
#include "src/ops/validation_internal.h"

namespace inferx::ops {

absl::Status ValidateGemm(const GemmRequest& request) {
  absl::Status status = internal::ValidateTensor(request.input, DType::kFloat32, 2, "gemm.input");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.weight, DType::kFloat32, 2, "gemm.weight");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.output, DType::kFloat32, 2, "gemm.output");
  if (!status.ok()) return status;
  const uint64_t rows = request.input.shape().dim(0);
  const uint64_t inner = request.input.shape().dim(1);
  const uint64_t columns = request.weight.shape().dim(0);
  if (inner == 0 || columns == 0) {
    return absl::InvalidArgumentError("gemm.shape: K and N must be positive");
  }
  if (request.weight.shape().dim(1) != inner || request.output.shape().dim(0) != rows ||
      request.output.shape().dim(1) != columns) {
    return absl::InvalidArgumentError("gemm.shape: expected A[M,K], B[N,K], D[M,N]");
  }
  if (!std::isfinite(request.alpha) || !std::isfinite(request.beta)) {
    return absl::InvalidArgumentError("gemm.scale: alpha and beta must be finite");
  }
  if (request.beta != 0.0F && !request.addend.has_value()) {
    return absl::InvalidArgumentError("gemm.addend: nonzero beta requires C");
  }
  if (request.addend.has_value()) {
    status = internal::ValidateTensor(*request.addend, DType::kFloat32, 2, "gemm.addend");
    if (!status.ok()) return status;
    if (request.addend->shape() != request.output.shape()) {
      return absl::InvalidArgumentError("gemm.addend: expected shape [M,N]");
    }
    status = internal::ValidateSameDevice(request.input, *request.addend, "gemm.device");
    if (!status.ok()) return status;
    absl::StatusOr<internal::TensorInterval> input = internal::Interval(request.input);
    absl::StatusOr<internal::TensorInterval> weight = internal::Interval(request.weight);
    absl::StatusOr<internal::TensorInterval> addend = internal::Interval(*request.addend);
    if (!input.ok()) return input.status();
    if (!weight.ok()) return weight.status();
    if (!addend.ok()) return addend.status();
    if (internal::Overlaps(*input, *addend) || internal::Overlaps(*weight, *addend)) {
      return absl::InvalidArgumentError("gemm.alias: addend overlaps input or weight");
    }
    status = internal::ValidateDisjoint(*request.addend, request.output, "gemm.alias");
    if (!status.ok()) return status;
  }
  status = internal::ValidateSameDevice(request.input, request.weight, "gemm.device");
  if (!status.ok()) return status;
  status = internal::ValidateSameDevice(request.input, request.output, "gemm.device");
  if (!status.ok()) return status;
  absl::StatusOr<internal::TensorInterval> input = internal::Interval(request.input);
  if (!input.ok()) return input.status();
  absl::StatusOr<internal::TensorInterval> weight = internal::Interval(request.weight);
  if (!weight.ok()) return weight.status();
  if (internal::Overlaps(*input, *weight)) {
    return absl::InvalidArgumentError("gemm.alias: input and weight overlap");
  }
  status = internal::ValidateDisjoint(request.input, request.output, "gemm.alias");
  if (!status.ok()) return status;
  return internal::ValidateDisjoint(request.weight, request.output, "gemm.alias");
}

absl::Status ReferenceGemm(const GemmRequest& request) {
  absl::Status status = ValidateGemm(request);
  if (!status.ok()) return status;
  absl::StatusOr<std::span<const float>> input = internal::HostSpan<float>(request.input);
  if (!input.ok()) return input.status();
  absl::StatusOr<std::span<const float>> weight = internal::HostSpan<float>(request.weight);
  if (!weight.ok()) return weight.status();
  absl::StatusOr<std::span<float>> output = internal::HostSpan<float>(request.output);
  if (!output.ok()) return output.status();
  std::span<const float> addend;
  if (request.addend.has_value()) {
    absl::StatusOr<std::span<const float>> values = internal::HostSpan<float>(*request.addend);
    if (!values.ok()) return values.status();
    addend = *values;
  }

  const size_t rows = static_cast<size_t>(request.input.shape().dim(0));
  const size_t inner = static_cast<size_t>(request.input.shape().dim(1));
  const size_t columns = static_cast<size_t>(request.weight.shape().dim(0));
  for (size_t row = 0; row < rows; ++row) {
    for (size_t column = 0; column < columns; ++column) {
      float sum = 0.0F;
      for (size_t index = 0; index < inner; ++index) {
        sum += (*input)[row * inner + index] * (*weight)[column * inner + index];
      }
      const size_t output_index = row * columns + column;
      const float addend_value = addend.empty() ? 0.0F : addend[output_index];
      (*output)[output_index] = request.alpha * sum + request.beta * addend_value;
    }
  }
  return absl::OkStatus();
}

}  // namespace inferx::ops
