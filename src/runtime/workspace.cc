#include "inferx/runtime/workspace.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"

namespace inferx {

WorkspaceLease::WorkspaceLease(BufferLease lease) noexcept : lease_(std::move(lease)) {}

absl::StatusOr<MutableBufferView> WorkspaceLease::Allocate(ByteCount bytes, ByteCount alignment,
                                                           WorkspaceTag tag) {
  const size_t tag_index = static_cast<size_t>(tag);
  if (tag_index >= high_water_.size()) {
    return absl::InvalidArgumentError("workspace.tag: invalid workspace tag");
  }
  if (!lease_.active()) {
    return absl::FailedPreconditionError("workspace.allocate: lease is inactive");
  }
  if (alignment.value() == 0 || !std::has_single_bit(alignment.value())) {
    return absl::InvalidArgumentError("workspace.alignment: must be a nonzero power of two");
  }
  absl::StatusOr<uint64_t> padded =
      CheckedAdd(offset_.value(), alignment.value() - 1, "workspace.allocate");
  if (!padded.ok()) {
    return padded.status();
  }
  const uint64_t aligned_offset = *padded & ~(alignment.value() - uint64_t{1});
  absl::StatusOr<uint64_t> end = CheckedAdd(aligned_offset, bytes.value(), "workspace.allocate");
  if (!end.ok()) {
    return end.status();
  }
  absl::StatusOr<MutableBufferView> arena = lease_.mutable_view();
  if (!arena.ok()) {
    return arena.status();
  }
  if (*end > arena->size().value()) {
    return absl::ResourceExhaustedError("workspace.allocate: arena capacity exceeded");
  }
  absl::StatusOr<MutableBufferView> slice =
      arena->Subview(ByteRange{ByteCount(aligned_offset), bytes});
  if (!slice.ok()) {
    return slice.status();
  }
  offset_ = ByteCount(*end);
  if (offset_.value() > high_water_[tag_index].value()) {
    high_water_[tag_index] = offset_;
  }
  return *slice;
}

ByteCount WorkspaceLease::high_water(WorkspaceTag tag) const noexcept {
  const size_t index = static_cast<size_t>(tag);
  return index < high_water_.size() ? high_water_[index] : ByteCount(0);
}

absl::Status WorkspaceLease::Release() { return lease_.Release(); }

absl::StatusOr<WorkspaceArenaPool> WorkspaceArenaPool::Create(Buffer backing, uint32_t slot_count,
                                                              ByteCount bytes_per_slot,
                                                              ByteCount alignment) {
  PoolGeometry geometry{slot_count, bytes_per_slot, alignment, PoolGeneration(0)};
  absl::StatusOr<FixedBufferPool> pool = FixedBufferPool::Create(std::move(backing), geometry);
  if (!pool.ok()) {
    return pool.status();
  }
  return WorkspaceArenaPool(std::move(*pool));
}

WorkspaceArenaPool::WorkspaceArenaPool(FixedBufferPool pool) noexcept : pool_(std::move(pool)) {}

absl::StatusOr<WorkspaceLease> WorkspaceArenaPool::Acquire() {
  absl::StatusOr<BufferLease> lease = pool_.Acquire();
  if (!lease.ok()) {
    return lease.status();
  }
  return WorkspaceLease(std::move(*lease));
}

absl::Status WorkspaceArenaPool::Close() { return pool_.Close(); }

absl::Status WorkspaceArenaPool::ValidateInvariants() const { return pool_.ValidateInvariants(); }

}  // namespace inferx
