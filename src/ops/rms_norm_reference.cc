#include <cmath>
#include <cstddef>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/ops/rms_norm.h"
#include "src/ops/validation_internal.h"

namespace inferx::ops {

absl::Status ValidateRmsNorm(const RmsNormRequest& request) {
  absl::Status status =
      internal::ValidateTensor(request.input, Dtype::kFloat32, 2, "rms_norm.input");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.weight, Dtype::kFloat32, 1, "rms_norm.weight");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.output, Dtype::kFloat32, 2, "rms_norm.output");
  if (!status.ok()) return status;
  const uint64_t hidden = request.input.shape().dim(1);
  if (hidden == 0 || request.weight.shape().dim(0) != hidden ||
      request.output.shape() != request.input.shape()) {
    return absl::InvalidArgumentError("rms_norm.shape: expected x[T,H], weight[H], output[T,H]");
  }
  if (!std::isfinite(request.epsilon) || request.epsilon <= 0.0F) {
    return absl::InvalidArgumentError("rms_norm.epsilon: must be finite and positive");
  }
  status = internal::ValidateSameDevice(request.input, request.weight, "rms_norm.device");
  if (!status.ok()) return status;
  status = internal::ValidateSameDevice(request.input, request.output, "rms_norm.device");
  if (!status.ok()) return status;
  status = internal::ValidateDisjoint(request.input, request.output, "rms_norm.alias");
  if (!status.ok()) return status;
  return internal::ValidateDisjoint(request.weight, request.output, "rms_norm.alias");
}

absl::Status ReferenceRmsNorm(const RmsNormRequest& request) {
  absl::Status status = ValidateRmsNorm(request);
  if (!status.ok()) return status;
  absl::StatusOr<std::span<const float>> input = internal::HostSpan<float>(request.input);
  if (!input.ok()) return input.status();
  absl::StatusOr<std::span<const float>> weight = internal::HostSpan<float>(request.weight);
  if (!weight.ok()) return weight.status();
  absl::StatusOr<std::span<float>> output = internal::HostSpan<float>(request.output);
  if (!output.ok()) return output.status();
  const size_t tokens = static_cast<size_t>(request.input.shape().dim(0));
  const size_t hidden = static_cast<size_t>(request.input.shape().dim(1));
  for (size_t token = 0; token < tokens; ++token) {
    float sum = 0.0F;
    for (size_t index = 0; index < hidden; ++index) {
      const float value = (*input)[token * hidden + index];
      sum += value * value;
    }
    const float mean_square = sum / static_cast<float>(hidden);
    const float inverse_rms = 1.0F / std::sqrt(mean_square + request.epsilon);
    for (size_t index = 0; index < hidden; ++index) {
      (*output)[token * hidden + index] =
          (*input)[token * hidden + index] * inverse_rms * (*weight)[index];
    }
  }
  return absl::OkStatus();
}

}  // namespace inferx::ops
