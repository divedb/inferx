#include <cmath>
#include <cstddef>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/ops/swiglu.h"
#include "src/ops/validation_internal.h"

namespace inferx::ops {
namespace {

absl::Status ValidateElementwiseInput(const TensorView& input, const MutableTensorView& output,
                                      const char* field) {
  absl::Status status =
      internal::ValidateTensor(input, Dtype::kFloat32, input.shape().rank(), field);
  if (!status.ok()) return status;
  status = internal::ValidateTensor(output, Dtype::kFloat32, input.shape().rank(), field);
  if (!status.ok()) return status;
  if (input.shape().rank() == 0 || input.shape() != output.shape()) {
    return absl::InvalidArgumentError("elementwise.shape: positive equal-rank shapes are required");
  }
  return internal::ValidateSameDevice(input, output, "elementwise.device");
}

absl::Status ValidateExactOrDisjoint(const TensorView& input, const MutableTensorView& output,
                                     const char* field) {
  absl::StatusOr<internal::TensorInterval> input_interval = internal::Interval(input);
  if (!input_interval.ok()) return input_interval.status();
  absl::StatusOr<internal::TensorInterval> output_interval = internal::Interval(output);
  if (!output_interval.ok()) return output_interval.status();
  if (internal::Overlaps(*input_interval, *output_interval) &&
      !internal::ExactAlias(*input_interval, *output_interval)) {
    return absl::InvalidArgumentError(field);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateSilu(const SiluRequest& request) {
  absl::Status status = ValidateElementwiseInput(request.input, request.output, "silu.tensor");
  if (!status.ok()) return status;
  return ValidateExactOrDisjoint(request.input, request.output,
                                 "silu.alias: only exact in-place alias is allowed");
}

absl::Status ReferenceSilu(const SiluRequest& request) {
  absl::Status status = ValidateSilu(request);
  if (!status.ok()) return status;
  absl::StatusOr<std::span<const float>> input = internal::HostSpan<float>(request.input);
  if (!input.ok()) return input.status();
  absl::StatusOr<std::span<float>> output = internal::HostSpan<float>(request.output);
  if (!output.ok()) return output.status();
  for (size_t index = 0; index < input->size(); ++index) {
    const float value = (*input)[index];
    (*output)[index] = value / (1.0F + std::exp(-value));
  }
  return absl::OkStatus();
}

absl::Status ValidateMultiply(const MultiplyRequest& request) {
  absl::Status status = ValidateElementwiseInput(request.left, request.output, "multiply.left");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.right, Dtype::kFloat32, request.left.shape().rank(),
                                    "multiply.right");
  if (!status.ok()) return status;
  if (request.right.shape() != request.left.shape()) {
    return absl::InvalidArgumentError("multiply.shape: input shapes must be equal");
  }
  status = internal::ValidateSameDevice(request.left, request.right, "multiply.device");
  if (!status.ok()) return status;
  absl::StatusOr<internal::TensorInterval> left = internal::Interval(request.left);
  absl::StatusOr<internal::TensorInterval> right = internal::Interval(request.right);
  absl::StatusOr<internal::TensorInterval> output = internal::Interval(request.output);
  if (!left.ok()) return left.status();
  if (!right.ok()) return right.status();
  if (!output.ok()) return output.status();
  if (internal::Overlaps(*left, *output) && !internal::ExactAlias(*left, *output)) {
    return absl::InvalidArgumentError("multiply.alias: partial left/output overlap");
  }
  if (internal::Overlaps(*right, *output) && !internal::ExactAlias(*right, *output)) {
    return absl::InvalidArgumentError("multiply.alias: partial right/output overlap");
  }
  if (internal::ExactAlias(*left, *output) && internal::Overlaps(*right, *output)) {
    return absl::InvalidArgumentError("multiply.alias: in-place left output overlaps right input");
  }
  if (internal::ExactAlias(*right, *output) && internal::Overlaps(*left, *output)) {
    return absl::InvalidArgumentError("multiply.alias: in-place right output overlaps left input");
  }
  return absl::OkStatus();
}

absl::Status ReferenceMultiply(const MultiplyRequest& request) {
  absl::Status status = ValidateMultiply(request);
  if (!status.ok()) return status;
  absl::StatusOr<std::span<const float>> left = internal::HostSpan<float>(request.left);
  if (!left.ok()) return left.status();
  absl::StatusOr<std::span<const float>> right = internal::HostSpan<float>(request.right);
  if (!right.ok()) return right.status();
  absl::StatusOr<std::span<float>> output = internal::HostSpan<float>(request.output);
  if (!output.ok()) return output.status();
  for (size_t index = 0; index < left->size(); ++index) {
    (*output)[index] = (*left)[index] * (*right)[index];
  }
  return absl::OkStatus();
}

absl::Status ValidateSwiGlu(const SwiGluRequest& request) { return ValidateMultiply(request); }

absl::Status ReferenceSwiGlu(const SwiGluRequest& request) {
  absl::Status status = ValidateSwiGlu(request);
  if (!status.ok()) return status;
  absl::StatusOr<std::span<const float>> gate = internal::HostSpan<float>(request.left);
  if (!gate.ok()) return gate.status();
  absl::StatusOr<std::span<const float>> up = internal::HostSpan<float>(request.right);
  if (!up.ok()) return up.status();
  absl::StatusOr<std::span<float>> output = internal::HostSpan<float>(request.output);
  if (!output.ok()) return output.status();
  for (size_t index = 0; index < gate->size(); ++index) {
    const float value = (*gate)[index];
    (*output)[index] = (value / (1.0F + std::exp(-value))) * (*up)[index];
  }
  return absl::OkStatus();
}

absl::Status ValidateResidual(const ResidualRequest& request) {
  absl::Status status = ValidateElementwiseInput(request.left, request.output, "residual.left");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.right, Dtype::kFloat32, request.left.shape().rank(),
                                    "residual.right");
  if (!status.ok()) return status;
  if (request.right.shape() != request.left.shape()) {
    return absl::InvalidArgumentError("residual.shape: input shapes must be equal");
  }
  status = internal::ValidateSameDevice(request.left, request.right, "residual.device");
  if (!status.ok()) return status;
  status = ValidateExactOrDisjoint(request.left, request.output,
                                   "residual.alias: partial left/output overlap");
  if (!status.ok()) return status;
  return ValidateExactOrDisjoint(request.right, request.output,
                                 "residual.alias: partial right/output overlap");
}

absl::Status ReferenceResidual(const ResidualRequest& request) {
  absl::Status status = ValidateResidual(request);
  if (!status.ok()) return status;
  absl::StatusOr<std::span<const float>> left = internal::HostSpan<float>(request.left);
  if (!left.ok()) return left.status();
  absl::StatusOr<std::span<const float>> right = internal::HostSpan<float>(request.right);
  if (!right.ok()) return right.status();
  absl::StatusOr<std::span<float>> output = internal::HostSpan<float>(request.output);
  if (!output.ok()) return output.status();
  for (size_t index = 0; index < left->size(); ++index) {
    (*output)[index] = (*left)[index] + (*right)[index];
  }
  return absl::OkStatus();
}

}  // namespace inferx::ops
