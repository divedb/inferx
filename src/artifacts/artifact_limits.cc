#include "inferx/artifacts/artifact_limits.h"

#include "absl/status/status.h"

namespace inferx::artifacts {

absl::Status ArtifactLimits::Validate() const {
  if (max_json_bytes == 0 || max_safetensors_header_bytes == 0 || max_json_depth == 0 ||
      max_tensor_rank == 0 || max_tensors == 0 || max_shards == 0 || max_active_mappings == 0 ||
      max_mapped_bytes == 0 || map_window_bytes == 0) {
    return absl::InvalidArgumentError("artifact limits must all be greater than zero");
  }
  if (map_window_bytes > max_mapped_bytes) {
    return absl::InvalidArgumentError(
        "artifact.map_window_bytes exceeds artifact.max_mapped_bytes");
  }
  return absl::OkStatus();
}

}  // namespace inferx::artifacts
