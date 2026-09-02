#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/ops/rope.h"
#include "src/ops/validation_internal.h"

namespace inferx::ops {
namespace {

absl::Status ValidateOutputAlias(const TensorView& input, const MutableTensorView& output,
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

absl::Status ValidateRope(const RopeRequest& request) {
  absl::Status status = internal::ValidateTensor(request.query, DType::kFloat32, 3, "rope.query");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.key, DType::kFloat32, 3, "rope.key");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.positions, DType::kInt32, 1, "rope.positions");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.query_output, DType::kFloat32, 3, "rope.query_output");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.key_output, DType::kFloat32, 3, "rope.key_output");
  if (!status.ok()) return status;
  const uint64_t tokens = request.query.shape().dim(0);
  const uint64_t head_dimension = request.query.shape().dim(2);
  if (head_dimension == 0 || head_dimension % 2 != 0 ||
      request.key.shape().dim(2) != head_dimension || request.key.shape().dim(0) != tokens ||
      request.positions.shape().dim(0) != tokens ||
      request.query_output.shape() != request.query.shape() ||
      request.key_output.shape() != request.key.shape()) {
    return absl::InvalidArgumentError(
        "rope.shape: expected Q[T,Nq,D], K[T,Nkv,D], positions[T] with positive even D");
  }
  if (!request.host_positions.empty() && request.host_positions.size() != tokens) {
    return absl::InvalidArgumentError("rope.host_positions: token count mismatch");
  }
  if (!std::isfinite(request.theta) || request.theta <= 0.0F ||
      request.max_position_embeddings == 0) {
    return absl::InvalidArgumentError("rope.parameters: theta and max position must be positive");
  }
  status = internal::ValidateSameDevice(request.query, request.key, "rope.device");
  if (!status.ok()) return status;
  status = internal::ValidateSameDevice(request.query, request.positions, "rope.device");
  if (!status.ok()) return status;
  status = internal::ValidateSameDevice(request.query, request.query_output, "rope.device");
  if (!status.ok()) return status;
  status = internal::ValidateSameDevice(request.query, request.key_output, "rope.device");
  if (!status.ok()) return status;
  status = ValidateOutputAlias(request.query, request.query_output,
                               "rope.alias: query output partially overlaps query input");
  if (!status.ok()) return status;
  status = ValidateOutputAlias(request.key, request.key_output,
                               "rope.alias: key output partially overlaps key input");
  if (!status.ok()) return status;
  absl::StatusOr<internal::TensorInterval> query = internal::Interval(request.query);
  absl::StatusOr<internal::TensorInterval> key = internal::Interval(request.key);
  absl::StatusOr<internal::TensorInterval> query_output = internal::Interval(request.query_output);
  absl::StatusOr<internal::TensorInterval> key_output = internal::Interval(request.key_output);
  if (!query.ok()) return query.status();
  if (!key.ok()) return key.status();
  if (!query_output.ok()) return query_output.status();
  if (!key_output.ok()) return key_output.status();
  if (internal::Overlaps(*query, *key) || internal::Overlaps(*query_output, *key) ||
      internal::Overlaps(*query, *key_output) || internal::Overlaps(*query_output, *key_output)) {
    return absl::InvalidArgumentError("rope.alias: query and key ranges must not overlap");
  }
  status = internal::ValidateDisjoint(request.positions, request.query_output, "rope.alias");
  if (!status.ok()) return status;
  return internal::ValidateDisjoint(request.positions, request.key_output, "rope.alias");
}

absl::Status ReferenceRope(const RopeRequest& request) {
  absl::Status status = ValidateRope(request);
  if (!status.ok()) return status;
  absl::StatusOr<std::span<const float>> query = internal::HostSpan<float>(request.query);
  if (!query.ok()) return query.status();
  absl::StatusOr<std::span<const float>> key = internal::HostSpan<float>(request.key);
  if (!key.ok()) return key.status();
  absl::StatusOr<std::span<const int32_t>> positions =
      internal::HostSpan<int32_t>(request.positions);
  if (!positions.ok()) return positions.status();
  absl::StatusOr<std::span<float>> query_output = internal::HostSpan<float>(request.query_output);
  if (!query_output.ok()) return query_output.status();
  absl::StatusOr<std::span<float>> key_output = internal::HostSpan<float>(request.key_output);
  if (!key_output.ok()) return key_output.status();

  for (size_t index = 0; index < positions->size(); ++index) {
    const int32_t position = (*positions)[index];
    if (position < 0 || static_cast<uint64_t>(position) >= request.max_position_embeddings ||
        (!request.host_positions.empty() && request.host_positions[index] != position)) {
      return absl::InvalidArgumentError(
          "rope.positions: position is outside model range or differs from host mirror");
    }
  }
  const size_t tokens = static_cast<size_t>(request.query.shape().dim(0));
  const size_t query_heads = static_cast<size_t>(request.query.shape().dim(1));
  const size_t key_heads = static_cast<size_t>(request.key.shape().dim(1));
  const size_t head_dimension = static_cast<size_t>(request.query.shape().dim(2));
  const size_t half = head_dimension / 2;
  auto rotate = [&](std::span<const float> input, std::span<float> output, size_t heads) {
    for (size_t token = 0; token < tokens; ++token) {
      for (size_t head = 0; head < heads; ++head) {
        const size_t base = (token * heads + head) * head_dimension;
        for (size_t index = 0; index < half; ++index) {
          const float exponent =
              -2.0F * static_cast<float>(index) / static_cast<float>(head_dimension);
          const float angle =
              static_cast<float>((*positions)[token]) * std::pow(request.theta, exponent);
          const float cosine = std::cos(angle);
          const float sine = std::sin(angle);
          const float first = input[base + index];
          const float second = input[base + index + half];
          output[base + index] = first * cosine - second * sine;
          output[base + index + half] = second * cosine + first * sine;
        }
      }
    }
  };
  rotate(*query, *query_output, query_heads);
  rotate(*key, *key_output, key_heads);
  return absl::OkStatus();
}

}  // namespace inferx::ops
