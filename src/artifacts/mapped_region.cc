#include "inferx/artifacts/mapped_region.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace inferx::artifacts {

struct ShardMappingPool::Budget {
  mutable std::mutex mutex;
  uint64_t regions = 0;
  uint64_t bytes = 0;
};

struct MappedTensorLease::State {
  std::shared_ptr<ShardMappingPool::Budget> budget;
  void* mapping = nullptr;
  size_t mapping_size = 0;

  ~State() {
    if (mapping != nullptr && mapping_size != 0) {
      ::munmap(mapping, mapping_size);
      std::lock_guard lock(budget->mutex);
      --budget->regions;
      budget->bytes -= mapping_size;
    }
  }
};

MappedTensorLease::MappedTensorLease(std::shared_ptr<State> state, ArtifactByteRange requested,
                                     FileIdentity identity, const std::byte* data, size_t size)
    : state_(std::move(state)),
      requested_(requested),
      identity_(identity),
      data_(data),
      size_(size) {}

MappedTensorLease::~MappedTensorLease() = default;
MappedTensorLease::MappedTensorLease(MappedTensorLease&&) noexcept = default;
MappedTensorLease& MappedTensorLease::operator=(MappedTensorLease&&) noexcept = default;

std::span<const std::byte> MappedTensorLease::bytes() const {
  return std::span<const std::byte>(data_, size_);
}

ShardMappingPool::ShardMappingPool(ArtifactLimits limits)
    : budget_(std::make_shared<Budget>()), limits_(limits) {}

ShardMappingPool::ShardMappingPool(std::shared_ptr<Budget> budget, ArtifactLimits limits)
    : budget_(std::move(budget)), limits_(limits) {}

absl::StatusOr<MappedTensorLease> ShardMappingPool::Map(const ArtifactFile& file,
                                                        ArtifactByteRange range) {
  if (auto status = limits_.Validate(); !status.ok()) return status;
  if (range.offset > file.identity().size || range.size > file.identity().size - range.offset) {
    return absl::InvalidArgumentError("mapping range exceeds artifact file");
  }
  if (range.size > limits_.map_window_bytes) {
    return absl::ResourceExhaustedError("mapping range exceeds artifact.map_window_bytes");
  }
  if (auto status = file.CheckUnchanged(); !status.ok()) return status;
  if (range.size == 0) {
    return MappedTensorLease(nullptr, range, file.identity(), nullptr, 0);
  }
  if (range.size > std::numeric_limits<size_t>::max()) {
    return absl::ResourceExhaustedError("mapping range does not fit host address space");
  }

  const long page_result = ::sysconf(_SC_PAGESIZE);
  if (page_result <= 0) {
    return absl::InternalError("sysconf(_SC_PAGESIZE) failed");
  }
  const uint64_t page = static_cast<uint64_t>(page_result);
  const uint64_t aligned_offset = (range.offset / page) * page;
  const uint64_t prefix = range.offset - aligned_offset;
  if (range.size > std::numeric_limits<uint64_t>::max() - prefix) {
    return absl::OutOfRangeError("rounded mapping size overflows uint64");
  }
  const uint64_t needed = prefix + range.size;
  if (needed > std::numeric_limits<uint64_t>::max() - (page - 1)) {
    return absl::OutOfRangeError("page-rounded mapping size overflows uint64");
  }
  const uint64_t rounded = ((needed + page - 1) / page) * page;
  if (rounded > std::numeric_limits<size_t>::max() ||
      aligned_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
    return absl::ResourceExhaustedError("page-rounded mapping does not fit the host mmap ABI");
  }

  {
    std::lock_guard lock(budget_->mutex);
    if (budget_->regions >= limits_.max_active_mappings ||
        rounded > limits_.max_mapped_bytes - budget_->bytes) {
      return absl::ResourceExhaustedError(
          absl::StrCat("mapping budget exhausted: active_regions=", budget_->regions,
                       " active_bytes=", budget_->bytes));
    }
    ++budget_->regions;
    budget_->bytes += rounded;
  }

  void* mapping = ::mmap(nullptr, static_cast<size_t>(rounded), PROT_READ, MAP_PRIVATE,
                         file.native_fd(), static_cast<off_t>(aligned_offset));
  if (mapping == MAP_FAILED) {
    const int error = errno;
    std::lock_guard lock(budget_->mutex);
    --budget_->regions;
    budget_->bytes -= rounded;
    return absl::InternalError(absl::StrCat("mmap failed (errno=", error, ")"));
  }

  auto state = std::make_shared<MappedTensorLease::State>();
  state->budget = budget_;
  state->mapping = mapping;
  state->mapping_size = static_cast<size_t>(rounded);
  if (auto status = file.CheckUnchanged(); !status.ok()) return status;
  const auto* data = static_cast<const std::byte*>(mapping) + prefix;
  return MappedTensorLease(std::move(state), range, file.identity(), data,
                           static_cast<size_t>(range.size));
}

absl::StatusOr<MappedTensorReader> ShardMappingPool::OpenReader(const ArtifactFile& file,
                                                                ArtifactByteRange range) const {
  if (auto status = limits_.Validate(); !status.ok()) return status;
  if (range.offset > file.identity().size || range.size > file.identity().size - range.offset) {
    return absl::InvalidArgumentError("reader range exceeds artifact file");
  }
  auto duplicate = file.Duplicate();
  if (!duplicate.ok()) return duplicate.status();
  return MappedTensorReader(budget_, limits_, std::move(*duplicate), range);
}

uint64_t ShardMappingPool::active_regions() const {
  std::lock_guard lock(budget_->mutex);
  return budget_->regions;
}

uint64_t ShardMappingPool::active_mapped_bytes() const {
  std::lock_guard lock(budget_->mutex);
  return budget_->bytes;
}

MappedTensorReader::MappedTensorReader(std::shared_ptr<ShardMappingPool::Budget> budget,
                                       ArtifactLimits limits, ArtifactFile file,
                                       ArtifactByteRange requested)
    : budget_(std::move(budget)), limits_(limits), file_(std::move(file)), requested_(requested) {}

absl::StatusOr<std::optional<MappedTensorLease>> MappedTensorReader::Next() {
  if (consumed_ == requested_.size) return std::optional<MappedTensorLease>{};
  if (requested_.offset > std::numeric_limits<uint64_t>::max() - consumed_) {
    return absl::OutOfRangeError("reader window offset overflows uint64");
  }
  const uint64_t window_offset = requested_.offset + consumed_;
  const long page_result = ::sysconf(_SC_PAGESIZE);
  if (page_result <= 0) {
    return absl::InternalError("sysconf(_SC_PAGESIZE) failed");
  }
  const uint64_t page = static_cast<uint64_t>(page_result);
  const uint64_t rounded_capacity = (limits_.max_mapped_bytes / page) * page;
  const uint64_t prefix = window_offset % page;
  if (rounded_capacity <= prefix) {
    return absl::ResourceExhaustedError(
        "artifact.max_mapped_bytes cannot hold one page-aligned reader window");
  }
  const uint64_t remaining = requested_.size - consumed_;
  const uint64_t payload_capacity = rounded_capacity - prefix;
  const uint64_t window_size = std::min({remaining, limits_.map_window_bytes, payload_capacity});
  const ArtifactByteRange window{window_offset, window_size};
  ShardMappingPool pool(budget_, limits_);
  auto lease = pool.Map(file_, window);
  if (!lease.ok()) return lease.status();
  consumed_ += window_size;
  return std::optional<MappedTensorLease>(std::move(*lease));
}

}  // namespace inferx::artifacts
