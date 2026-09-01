// Fixed device-neutral workspace arenas with checked bump allocation.
#ifndef INFERX_RUNTIME_WORKSPACE_H_
#define INFERX_RUNTIME_WORKSPACE_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/token.h"
#include "inferx/runtime/buffer_pool.h"

namespace inferx {

enum class WorkspaceTag : uint8_t {
  kKernelScratch = 0,
  kReduction = 1,
  kMetadata = 2,
  kTest = 3,
  kCount = 4,
};

class WorkspaceLease {
 public:
  WorkspaceLease() noexcept = default;
  WorkspaceLease(WorkspaceLease&&) noexcept = default;
  WorkspaceLease& operator=(WorkspaceLease&&) noexcept = default;
  WorkspaceLease(const WorkspaceLease&) = delete;
  WorkspaceLease& operator=(const WorkspaceLease&) = delete;

  [[nodiscard]] absl::StatusOr<MutableBufferView> Allocate(ByteCount bytes, ByteCount alignment,
                                                           WorkspaceTag tag);
  [[nodiscard]] ByteCount used() const noexcept { return offset_; }
  [[nodiscard]] ByteCount high_water(WorkspaceTag tag) const noexcept;
  [[nodiscard]] PoolToken token() const noexcept { return lease_.token(); }
  [[nodiscard]] absl::Status Release();

 private:
  friend class WorkspaceArenaPool;
  explicit WorkspaceLease(BufferLease lease) noexcept;

  BufferLease lease_;
  ByteCount offset_ = ByteCount(0);
  std::array<ByteCount, static_cast<size_t>(WorkspaceTag::kCount)> high_water_{
      ByteCount(0), ByteCount(0), ByteCount(0), ByteCount(0)};
};

class WorkspaceArenaPool {
 public:
  [[nodiscard]] static absl::StatusOr<WorkspaceArenaPool> Create(Buffer backing,
                                                                 uint32_t slot_count,
                                                                 ByteCount bytes_per_slot,
                                                                 ByteCount alignment);
  [[nodiscard]] absl::StatusOr<WorkspaceLease> Acquire();
  [[nodiscard]] absl::Status Close();
  [[nodiscard]] absl::Status ValidateInvariants() const;

 private:
  explicit WorkspaceArenaPool(FixedBufferPool pool) noexcept;
  FixedBufferPool pool_;
};

}  // namespace inferx

#endif  // INFERX_RUNTIME_WORKSPACE_H_
