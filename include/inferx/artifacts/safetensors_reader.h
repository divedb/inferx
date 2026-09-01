#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/artifacts/artifact_file.h"
#include "inferx/artifacts/artifact_limits.h"
#include "inferx/artifacts/artifact_tensor.h"

namespace inferx::artifacts {

struct SafeTensorsMetadata {
  uint64_t header_size = 0;
  uint64_t data_start = 0;
  uint64_t file_size = 0;
  std::vector<ArtifactTensor> tensors;
  std::map<std::string, std::string> user_metadata;

  const ArtifactTensor* Find(std::string_view name) const;
};

class SafeTensorReader {
 public:
  absl::StatusOr<SafeTensorsMetadata> ReadHeader(const ArtifactFile& file,
                                                 const ArtifactLimits& limits = {}) const;
};

}  // namespace inferx::artifacts
