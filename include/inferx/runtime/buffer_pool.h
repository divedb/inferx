// Fixed, generation-checked buffer suballocation.
#ifndef INFERX_RUNTIME_BUFFER_POOL_H_
#define INFERX_RUNTIME_BUFFER_POOL_H_

#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/tensor/buffer.h"

namespace inferx {

struct PoolGeometry {
  uint32_t slot_count = 0;
  ByteCount slot_bytes = ByteCount(0);
  ByteCount alignment = ByteCount(1);
  PoolGeneration initial_generation = PoolGeneration(0);
};

struct PoolToken {
  PoolId pool;
  PoolSlotId slot;
  PoolGeneration generation;

  friend constexpr bool operator==(const PoolToken&, const PoolToken&) = default;
};

class BufferPoolState;

class BufferLease {
 public:
  BufferLease() noexcept = default;
  ~BufferLease() noexcept;
  BufferLease(BufferLease&& other) noexcept;
  BufferLease& operator=(BufferLease&& other) noexcept;
  BufferLease(const BufferLease&) = delete;
  BufferLease& operator=(const BufferLease&) = delete;

  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] BufferView view() const noexcept {
    if (!view_.has_value()) {
      std::terminate();
    }
    return view_->AsConst();
  }
  [[nodiscard]] absl::StatusOr<MutableBufferView> mutable_view();
  [[nodiscard]] PoolToken token() const noexcept { return token_; }
  [[nodiscard]] absl::Status Release();

 private:
  friend class FixedBufferPool;
  friend class BufferPoolState;
  friend class MutexFixedBufferPool;
  BufferLease(std::shared_ptr<BufferPoolState> state, PoolToken token,
              MutableBufferView view) noexcept;

  std::shared_ptr<BufferPoolState> state_;
  PoolToken token_{PoolId(0), PoolSlotId(0), PoolGeneration(0)};
  std::optional<MutableBufferView> view_;
  std::mutex* adapter_mutex_ = nullptr;
  bool active_ = false;
};

class FixedBufferPool {
 public:
  [[nodiscard]] static absl::StatusOr<FixedBufferPool> Create(Buffer backing,
                                                              PoolGeometry geometry);

  FixedBufferPool(FixedBufferPool&& other) noexcept;
  FixedBufferPool& operator=(FixedBufferPool&& other) noexcept;
  FixedBufferPool(const FixedBufferPool&) = delete;
  FixedBufferPool& operator=(const FixedBufferPool&) = delete;
  ~FixedBufferPool() noexcept;

  [[nodiscard]] absl::StatusOr<BufferLease> Acquire();
  [[nodiscard]] absl::Status ValidateInvariants() const;
  [[nodiscard]] absl::Status Close();
  [[nodiscard]] uint32_t available_slots() const noexcept;
  [[nodiscard]] PoolGeometry geometry() const noexcept;

 private:
  explicit FixedBufferPool(std::shared_ptr<BufferPoolState> state) noexcept;
  std::shared_ptr<BufferPoolState> state_;
};

// Deliberate MPMC test adapter; production worker-owned pools use the
// lock-free-by-ownership FixedBufferPool directly.
class MutexFixedBufferPool {
 public:
  explicit MutexFixedBufferPool(FixedBufferPool pool) noexcept;
  [[nodiscard]] absl::StatusOr<BufferLease> Acquire();
  [[nodiscard]] absl::Status ValidateInvariants() const;

 private:
  mutable std::mutex mutex_;
  FixedBufferPool pool_;
};

}  // namespace inferx

#endif  // INFERX_RUNTIME_BUFFER_POOL_H_
