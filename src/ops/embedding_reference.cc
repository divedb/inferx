#include <algorithm>
#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/ops/embedding.h"
#include "src/ops/validation_internal.h"

namespace inferx::ops {

absl::Status ValidateEmbedding(const EmbeddingRequest& request) {
  absl::Status status =
      internal::ValidateTensor(request.token_ids, DType::kInt32, 1, "embedding.token_ids");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.weight, DType::kFloat32, 2, "embedding.weight");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.output, DType::kFloat32, 2, "embedding.output");
  if (!status.ok()) return status;
  if (request.weight.shape().dim(0) == 0 || request.weight.shape().dim(1) == 0) {
    return absl::InvalidArgumentError(
        "embedding.weight: vocabulary and hidden dimensions must be positive");
  }
  if (request.output.shape().dim(0) != request.token_ids.shape().dim(0) ||
      request.output.shape().dim(1) != request.weight.shape().dim(1)) {
    return absl::InvalidArgumentError("embedding.output: expected shape [tokens, hidden]");
  }
  if (!request.host_token_ids.empty() &&
      request.host_token_ids.size() != request.token_ids.shape().dim(0)) {
    return absl::InvalidArgumentError("embedding.host_token_ids: token count mismatch");
  }
  status = internal::ValidateSameDevice(request.token_ids, request.weight, "embedding.device");
  if (!status.ok()) return status;
  status = internal::ValidateSameDevice(request.token_ids, request.output, "embedding.device");
  if (!status.ok()) return status;
  absl::StatusOr<internal::TensorInterval> token_ids = internal::Interval(request.token_ids);
  if (!token_ids.ok()) return token_ids.status();
  absl::StatusOr<internal::TensorInterval> weight = internal::Interval(request.weight);
  if (!weight.ok()) return weight.status();
  if (internal::Overlaps(*token_ids, *weight)) {
    return absl::InvalidArgumentError("embedding.alias: IDs and weight overlap");
  }
  status = internal::ValidateDisjoint(request.token_ids, request.output, "embedding.alias");
  if (!status.ok()) return status;
  return internal::ValidateDisjoint(request.weight, request.output, "embedding.alias");
}

absl::Status ReferenceEmbedding(const EmbeddingRequest& request) {
  absl::Status status = ValidateEmbedding(request);
  if (!status.ok()) return status;
  absl::StatusOr<std::span<const int32_t>> ids = internal::HostSpan<int32_t>(request.token_ids);
  if (!ids.ok()) return ids.status();
  absl::StatusOr<std::span<const float>> weight = internal::HostSpan<float>(request.weight);
  if (!weight.ok()) return weight.status();
  absl::StatusOr<std::span<float>> output = internal::HostSpan<float>(request.output);
  if (!output.ok()) return output.status();

  const uint64_t vocabulary = request.weight.shape().dim(0);
  const size_t hidden = static_cast<size_t>(request.weight.shape().dim(1));
  for (size_t index = 0; index < ids->size(); ++index) {
    const int32_t id = (*ids)[index];
    if (id < 0 || static_cast<uint64_t>(id) >= vocabulary ||
        (!request.host_token_ids.empty() && request.host_token_ids[index] != id)) {
      return absl::InvalidArgumentError(
          "embedding.token_ids: token ID is outside vocabulary or differs from host mirror");
    }
  }
  for (size_t token = 0; token < ids->size(); ++token) {
    const size_t row = static_cast<size_t>((*ids)[token]);
    std::copy_n(weight->begin() + static_cast<std::ptrdiff_t>(row * hidden), hidden,
                output->begin() + static_cast<std::ptrdiff_t>(token * hidden));
  }
  return absl::OkStatus();
}

}  // namespace inferx::ops
