#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>

#include "absl/status/statusor.h"
#include "inferx/artifacts/artifact_file.h"
#include "inferx/artifacts/artifact_limits.h"
#include "inferx/artifacts/artifact_tensor.h"

namespace inferx::artifacts {

class ShardMappingPool;
class MappedTensorReader;

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
  explicit ShardMappingPool(ArtifactLimits limits);

  absl::StatusOr<MappedTensorLease> Map(const ArtifactFile& file, ArtifactByteRange range);
  absl::StatusOr<MappedTensorReader> OpenReader(const ArtifactFile& file,
                                                ArtifactByteRange range) const;
  uint64_t active_regions() const;
  uint64_t active_mapped_bytes() const;

 private:
  friend class MappedTensorLease;
  friend class MappedTensorReader;
  struct Budget;
  ShardMappingPool(std::shared_ptr<Budget> budget, ArtifactLimits limits);

  std::shared_ptr<Budget> budget_;
  ArtifactLimits limits_;
};

class MappedTensorReader {
 public:
  MappedTensorReader(MappedTensorReader&&) noexcept = default;
  MappedTensorReader& operator=(MappedTensorReader&&) noexcept = default;
  MappedTensorReader(const MappedTensorReader&) = delete;
  MappedTensorReader& operator=(const MappedTensorReader&) = delete;

  absl::StatusOr<std::optional<MappedTensorLease>> Next();
  ArtifactByteRange requested_range() const { return requested_; }
  uint64_t consumed_bytes() const { return consumed_; }
  uint64_t remaining_bytes() const { return requested_.size - consumed_; }

 private:
  friend class ShardMappingPool;
  MappedTensorReader(std::shared_ptr<ShardMappingPool::Budget> budget, ArtifactLimits limits,
                     ArtifactFile file, ArtifactByteRange requested);

  std::shared_ptr<ShardMappingPool::Budget> budget_;
  ArtifactLimits limits_;
  ArtifactFile file_;
  ArtifactByteRange requested_;
  uint64_t consumed_ = 0;
};

}  // namespace inferx::artifacts
