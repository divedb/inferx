#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/artifacts/artifact_file.h"
#include "inferx/artifacts/artifact_limits.h"
#include "inferx/artifacts/safe_relative_path.h"

namespace inferx::artifacts {

struct SafetensorsIndexEntry {
  std::string tensor_name;
  SafeRelativePath shard;
};

struct SafetensorsIndex {
  std::vector<SafetensorsIndexEntry> entries;
  std::vector<SafeRelativePath> shards;
  std::optional<uint64_t> declared_total_size;
};

class SafetensorsIndexReader {
 public:
  absl::StatusOr<SafetensorsIndex> Read(const ArtifactFile& file,
                                        const ArtifactLimits& limits = {}) const;
};

}  // namespace inferx::artifacts
