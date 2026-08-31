#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>

#include "absl/status/statusor.h"
#include "inferx/artifacts/artifact_file.h"
#include "inferx/artifacts/artifact_limits.h"
#include "inferx/artifacts/artifact_tensor.h"

namespace inferx::artifacts {

class ShardMappingPool;

class MappedTensorLease {
 public:
  MappedTensorLease() = default;
  ~MappedTensorLease();
  MappedTensorLease(MappedTensorLease&&) noexcept;
  MappedTensorLease& operator=(MappedTensorLease&&) noexcept;
  MappedTensorLease(const MappedTensorLease&) = delete;
  MappedTensorLease& operator=(const MappedTensorLease&) = delete;

  std::span<const std::byte> bytes() const;
  ArtifactByteRange requested_range() const { return requested_; }
  FileIdentity file_identity() const { return identity_; }

 private:
  friend class ShardMappingPool;
  struct State;
  MappedTensorLease(std::shared_ptr<State> state, ArtifactByteRange requested,
                    FileIdentity identity, const std::byte* data, size_t size);

  std::shared_ptr<State> state_;
  ArtifactByteRange requested_;
  FileIdentity identity_;
  const std::byte* data_ = nullptr;
  size_t size_ = 0;
};

class ShardMappingPool {
 public:
  struct Budget;

  explicit ShardMappingPool(ArtifactLimits limits);

  absl::StatusOr<MappedTensorLease> Map(const ArtifactFile& file, ArtifactByteRange range);
  uint64_t active_regions() const;
  uint64_t active_mapped_bytes() const;

 private:
  std::shared_ptr<Budget> budget_;
  ArtifactLimits limits_;
};

}  // namespace inferx::artifacts
