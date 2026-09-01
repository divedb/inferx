#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/artifacts/artifact_file.h"
#include "inferx/artifacts/artifact_limits.h"
#include "inferx/artifacts/digest.h"
#include "inferx/artifacts/safe_relative_path.h"

namespace inferx::artifacts {

struct ArtifactManifestEntry {
  SafeRelativePath path;
  uint64_t size = 0;
  Digest256 digest;
};

struct ArtifactManifest {
  uint32_t schema_version = 1;
  std::string model_revision;
  SafeRelativePath weights_entry;
  std::vector<ArtifactManifestEntry> files;

  const ArtifactManifestEntry* Find(const SafeRelativePath& path) const;
};

class ArtifactManifestReader {
 public:
  absl::StatusOr<ArtifactManifest> Read(const ArtifactFile& file,
                                        const ArtifactLimits& limits = {}) const;
};

absl::Status VerifyManifestEntry(const ArtifactManifestEntry& expected, const ArtifactFile& file,
                                 const Digest256& digest);

}  // namespace inferx::artifacts
