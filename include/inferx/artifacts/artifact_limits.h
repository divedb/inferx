#pragma once

#include <cstdint>

#include "absl/status/status.h"

namespace inferx::artifacts {

struct ArtifactLimits {
  uint64_t max_json_bytes = 16ULL * 1024 * 1024;
  uint64_t max_safetensors_header_bytes = 100'000'000;
  uint32_t max_json_depth = 64;
  uint32_t max_tensor_rank = 8;
  uint64_t max_tensors = 1'000'000;
  uint64_t max_shards = 100'000;
  uint64_t max_active_mappings = 32;
  uint64_t max_mapped_bytes = 2ULL * 1024 * 1024 * 1024;
  uint64_t map_window_bytes = 256ULL * 1024 * 1024;

  absl::Status Validate() const;
};

}  // namespace inferx::artifacts
