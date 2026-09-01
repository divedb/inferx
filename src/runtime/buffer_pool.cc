#include "inferx/runtime/buffer_pool.h"

#include <atomic>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"

namespace inferx {
namespace {

enum class SlotState : uint8_t { kFree, kLeased, kRetired };

struct Slot {
  SlotState state = SlotState::kFree;
  PoolGeneration generation = PoolGeneration(0);
};

std::atomic<uint64_t> g_next_pool_id{1};

absl::Status PoolError(absl::StatusCode code, ErrorReason reason, absl::string_view message) {
  return WithErrorReason(absl::Status(code, message), reason);
}

}  // namespace

class BufferPoolState {
 public:
  BufferPoolState(Buffer backing_value, PoolGeometry geometry_value, uint64_t slot_stride_value,
                  PoolId id_value, std::vector<Slot> slots_value)
      : backing(std::move(backing_value)),
        geometry(geometry_value),
        slot_stride(slot_stride_value),
        id(id_value),
        slots(std::move(slots_value)) {}

  absl::StatusOr<BufferLease> Acquire() {
    if (closed) {
      return absl::FailedPreconditionError("buffer_pool.acquire: pool is closed");
    }
    for (uint32_t index = 0; index < geometry.slot_count; ++index) {
      Slot& slot = slots[index];
      if (slot.state != SlotState::kFree) {
        continue;
      }
      if (slot.generation.value() == std::numeric_limits<uint64_t>::max()) {
        slot.state = SlotState::kRetired;
        continue;
      }
      slot.generation = PoolGeneration(slot.generation.value() + 1);
      const uint64_t offset = static_cast<uint64_t>(index) * slot_stride;
      absl::StatusOr<MutableBufferView> view =
          backing.MutableView(ByteRange{ByteCount(offset), geometry.slot_bytes});
      if (!view.ok()) {
        return view.status();
      }
      slot.state = SlotState::kLeased;
      ++live_leases;
      PoolToken token{id, PoolSlotId(index), slot.generation};
      return BufferLease(shared_from_this(), token, *view);
    }
    return PoolError(absl::StatusCode::kResourceExhausted, ErrorReason::kPoolExhausted,
                     "buffer_pool.acquire: no free slots");
  }

  absl::Status Release(PoolToken token) {
    if (token.pool != id || token.slot.value() >= slots.size()) {
      return PoolError(absl::StatusCode::kFailedPrecondition, ErrorReason::kStalePoolLease,
                       "buffer_pool.release: foreign pool token");
    }
    Slot& slot = slots[token.slot.value()];
    if (slot.state != SlotState::kLeased || slot.generation != token.generation) {
      return PoolError(absl::StatusCode::kFailedPrecondition, ErrorReason::kStalePoolLease,
                       "buffer_pool.release: duplicate or stale token");
    }
    slot.state = SlotState::kFree;
    --live_leases;
    return absl::OkStatus();
  }

  absl::Status ValidateInvariants() const {
    uint32_t counted_live = 0;
    for (const Slot& slot : slots) {
      if (slot.state == SlotState::kLeased) {
        ++counted_live;
      }
    }
    if (counted_live != live_leases) {
      return PoolError(absl::StatusCode::kInternal, ErrorReason::kInvariantViolation,
                       "buffer_pool: lease count invariant failed");
    }
    return absl::OkStatus();
  }

  absl::Status Close() {
    if (closed) {
      return absl::OkStatus();
    }
    if (live_leases != 0) {
      return PoolError(absl::StatusCode::kFailedPrecondition, ErrorReason::kPendingResource,
                       "buffer_pool.close: live leases remain");
    }
    absl::Status status = backing.Release();
    if (status.ok()) {
      closed = true;
    }
    return status;
  }

  uint32_t available_slots() const noexcept {
    uint32_t available = 0;
    for (const Slot& slot : slots) {
      if (slot.state == SlotState::kFree) {
        ++available;
      }
    }
    return available;
  }

  Buffer backing;
  PoolGeometry geometry;
  uint64_t slot_stride;
  PoolId id;
  std::vector<Slot> slots;
  uint32_t live_leases = 0;
  bool closed = false;

 private:
  // Enable shared ownership without exposing it in the installed contract.
  struct EnableShared : public std::enable_shared_from_this<BufferPoolState> {};
  std::shared_ptr<BufferPoolState> shared_from_this() { return self.lock(); }

 public:
  std::weak_ptr<BufferPoolState> self;
};

BufferLease::BufferLease(std::shared_ptr<BufferPoolState> state, PoolToken token,
                         MutableBufferView view) noexcept
    : state_(std::move(state)), token_(token), view_(view), active_(true) {}

BufferLease::~BufferLease() noexcept {
  if (active_) {
    static_cast<void>(Release());
  }
}

BufferLease::BufferLease(BufferLease&& other) noexcept
    : state_(std::move(other.state_)),
      token_(other.token_),
      view_(other.view_),
      adapter_mutex_(std::exchange(other.adapter_mutex_, nullptr)),
      active_(std::exchange(other.active_, false)) {}

BufferLease& BufferLease::operator=(BufferLease&& other) noexcept {
  if (this != &other) {
    if (active_) {
      static_cast<void>(Release());
    }
    state_ = std::move(other.state_);
    token_ = other.token_;
    view_ = other.view_;
    adapter_mutex_ = std::exchange(other.adapter_mutex_, nullptr);
    active_ = std::exchange(other.active_, false);
  }
  return *this;
}

absl::StatusOr<MutableBufferView> BufferLease::mutable_view() {
  if (!active_ || !view_.has_value()) {
    return absl::FailedPreconditionError("buffer_lease.view: lease is inactive");
  }
  return *view_;
}

absl::Status BufferLease::Release() {
  if (!active_ || state_ == nullptr) {
    return PoolError(absl::StatusCode::kFailedPrecondition, ErrorReason::kStalePoolLease,
                     "buffer_lease.release: lease is inactive");
  }
  std::unique_lock<std::mutex> lock;
  if (adapter_mutex_ != nullptr) {
    lock = std::unique_lock<std::mutex>(*adapter_mutex_);
  }
  absl::Status status = state_->Release(token_);
  if (status.ok()) {
    active_ = false;
    view_.reset();
    state_.reset();
    adapter_mutex_ = nullptr;
  }
  return status;
}

absl::StatusOr<FixedBufferPool> FixedBufferPool::Create(Buffer backing, PoolGeometry geometry) {
  if (backing.empty()) {
    return absl::InvalidArgumentError("buffer_pool.backing: buffer is released");
  }
  const auto reject = [&backing](absl::Status status) -> absl::StatusOr<FixedBufferPool> {
    static_cast<void>(backing.Release());
    return status;
  };
  if (geometry.slot_count == 0 || geometry.slot_bytes.value() == 0) {
    return reject(
        absl::InvalidArgumentError("buffer_pool.geometry: slot count and size must be positive"));
  }
  if (geometry.alignment.value() == 0 || !std::has_single_bit(geometry.alignment.value())) {
    return reject(
        absl::InvalidArgumentError("buffer_pool.alignment: must be a nonzero power of two"));
  }
  if (backing.alignment().value() < geometry.alignment.value()) {
    return reject(
        absl::InvalidArgumentError("buffer_pool.backing: insufficient backing alignment"));
  }
  absl::StatusOr<uint64_t> padded = CheckedAdd(
      geometry.slot_bytes.value(), geometry.alignment.value() - 1, "buffer_pool.slot_stride");
  if (!padded.ok()) {
    return reject(padded.status());
  }
  const uint64_t stride = *padded & ~(geometry.alignment.value() - uint64_t{1});
  absl::StatusOr<uint64_t> required =
      CheckedMul(stride, static_cast<uint64_t>(geometry.slot_count), "buffer_pool.required_bytes");
  if (!required.ok()) {
    return reject(required.status());
  }
  if (*required > backing.size().value()) {
    return reject(
        absl::OutOfRangeError("buffer_pool.backing: allocation is smaller than pool geometry"));
  }
  uint64_t numeric_id = g_next_pool_id.load(std::memory_order_relaxed);
  while (numeric_id != std::numeric_limits<uint64_t>::max() &&
         !g_next_pool_id.compare_exchange_weak(
             numeric_id, numeric_id + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
  if (numeric_id == std::numeric_limits<uint64_t>::max()) {
    return reject(absl::OutOfRangeError("buffer_pool.id: pool ID exhausted"));
  }
  std::vector<Slot> slots;
  try {
    slots.assign(geometry.slot_count, Slot{SlotState::kFree, geometry.initial_generation});
  } catch (const std::bad_alloc&) {
    return reject(absl::ResourceExhaustedError("buffer_pool: slot table allocation failed"));
  }
  std::shared_ptr<BufferPoolState> state;
  try {
    state = std::make_shared<BufferPoolState>(std::move(backing), geometry, stride,
                                              PoolId(numeric_id), std::move(slots));
  } catch (const std::bad_alloc&) {
    return reject(absl::ResourceExhaustedError("buffer_pool: pool state allocation failed"));
  }
  state->self = state;
  return FixedBufferPool(std::move(state));
}

FixedBufferPool::FixedBufferPool(std::shared_ptr<BufferPoolState> state) noexcept
    : state_(std::move(state)) {}

FixedBufferPool::FixedBufferPool(FixedBufferPool&& other) noexcept
    : state_(std::move(other.state_)) {}

FixedBufferPool& FixedBufferPool::operator=(FixedBufferPool&& other) noexcept {
  if (this != &other) {
    if (state_ != nullptr && !state_->Close().ok()) {
      std::terminate();
    }
    state_ = std::move(other.state_);
  }
  return *this;
}

FixedBufferPool::~FixedBufferPool() noexcept {
  if (state_ != nullptr && !state_->Close().ok()) {
    std::terminate();
  }
}

absl::StatusOr<BufferLease> FixedBufferPool::Acquire() {
  if (state_ == nullptr) {
    return absl::FailedPreconditionError("buffer_pool.acquire: pool was moved");
  }
  return state_->Acquire();
}

absl::Status FixedBufferPool::ValidateInvariants() const {
  if (state_ == nullptr) {
    return absl::FailedPreconditionError("buffer_pool.validate: pool was moved");
  }
  return state_->ValidateInvariants();
}

absl::Status FixedBufferPool::Close() {
  if (state_ == nullptr) {
    return absl::OkStatus();
  }
  absl::Status status = state_->Close();
  if (status.ok()) {
    state_.reset();
  }
  return status;
}

uint32_t FixedBufferPool::available_slots() const noexcept {
  return state_ == nullptr ? 0 : state_->available_slots();
}

PoolGeometry FixedBufferPool::geometry() const noexcept {
  return state_ == nullptr ? PoolGeometry{} : state_->geometry;
}

MutexFixedBufferPool::MutexFixedBufferPool(FixedBufferPool pool) noexcept
    : pool_(std::move(pool)) {}

absl::StatusOr<BufferLease> MutexFixedBufferPool::Acquire() {
  std::lock_guard<std::mutex> lock(mutex_);
  absl::StatusOr<BufferLease> lease = pool_.Acquire();
  if (lease.ok()) {
    lease->adapter_mutex_ = &mutex_;
  }
  return lease;
}

absl::Status MutexFixedBufferPool::ValidateInvariants() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pool_.ValidateInvariants();
}

}  // namespace inferx
