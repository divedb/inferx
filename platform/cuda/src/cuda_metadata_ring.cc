#include "inferx/platform/cuda/cuda_metadata_ring.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "inferx/base/status.h"
#include "inferx/platform/cuda/cuda_copy.h"

namespace inferx::cuda {
namespace {

enum class MetadataState : uint8_t { kFree, kHostWritable, kInFlight, kRetired };

struct MetadataSlot {
  Buffer pinned;
  Buffer device;
  PoolGeneration generation = PoolGeneration(0);
  MetadataState state = MetadataState::kFree;
  std::optional<CudaEventLease> upload_event;
};

absl::Status StaleMetadata(absl::string_view operation) {
  return WithErrorReason(absl::FailedPreconditionError(
                             absl::StrCat("cuda.metadata.", operation, ": stale metadata token")),
                         ErrorReason::kStalePoolLease);
}

}  // namespace

class MetadataRingState {
 public:
  absl::StatusOr<MetadataSlot*> Resolve(MetadataToken token) {
    if (token.slot.value() >= slots.size()) return StaleMetadata("resolve");
    MetadataSlot& slot = slots[token.slot.value()];
    if (slot.generation != token.generation || slot.state == MetadataState::kFree ||
        slot.state == MetadataState::kRetired) {
      return StaleMetadata("resolve");
    }
    return &slot;
  }

  CudaEventPool* event_pool = nullptr;
  const CudaApi* api = nullptr;
  CudaHealth* health = nullptr;
  DeviceId device = DeviceId(0);
  ByteCount slot_bytes = ByteCount(0);
  std::vector<MetadataSlot> slots;
  bool closed = false;
};

absl::Status ReleaseSlotBuffers(MetadataRingState& state) {
  absl::Status first_error = absl::OkStatus();
  for (MetadataSlot& slot : state.slots) {
    if (!slot.device.empty()) {
      absl::Status status = slot.device.Release();
      if (!status.ok() && first_error.ok()) first_error = status;
    }
    if (!slot.pinned.empty()) {
      absl::Status status = slot.pinned.Release();
      if (!status.ok() && first_error.ok()) first_error = status;
    }
  }
  return first_error;
}

CudaMetadataLease::CudaMetadataLease(std::shared_ptr<MetadataRingState> state,
                                     MetadataToken token) noexcept
    : state_(std::move(state)), token_(token) {}

CudaMetadataLease::~CudaMetadataLease() noexcept {
  // A sealed lease cannot safely release itself: its consumer fence is owned
  // elsewhere. Keeping the shared ring state preserves all backing storage.
  if (state_ != nullptr && !sealed_) static_cast<void>(Release());
}

CudaMetadataLease::CudaMetadataLease(CudaMetadataLease&& other) noexcept
    : state_(std::move(other.state_)),
      token_(other.token_),
      sealed_(std::exchange(other.sealed_, false)) {}

CudaMetadataLease& CudaMetadataLease::operator=(CudaMetadataLease&& other) noexcept {
  if (this != &other) {
    if (state_ != nullptr && !sealed_) static_cast<void>(Release());
    state_ = std::move(other.state_);
    token_ = other.token_;
    sealed_ = std::exchange(other.sealed_, false);
  }
  return *this;
}

absl::StatusOr<MutableBufferView> CudaMetadataLease::host_view() {
  if (state_ == nullptr || sealed_) {
    return absl::FailedPreconditionError("cuda.metadata.host_view: lease is inactive or sealed");
  }
  absl::StatusOr<MetadataSlot*> slot = state_->Resolve(token_);
  if (!slot.ok()) return slot.status();
  if ((*slot)->state != MetadataState::kHostWritable) {
    return absl::FailedPreconditionError("cuda.metadata.host_view: slot is not host writable");
  }
  return (*slot)->pinned.MutableView(ByteRange{ByteCount(0), state_->slot_bytes});
}

absl::StatusOr<BufferView> CudaMetadataLease::SealAndUpload(CudaStream& transfer,
                                                            CudaStream& compute) {
  if (state_ == nullptr || sealed_) {
    return absl::FailedPreconditionError(
        "cuda.metadata.upload: lease is inactive or already sealed");
  }
  if (transfer.device() != state_->device || compute.device() != state_->device) {
    return absl::InvalidArgumentError(
        "cuda.metadata.upload: streams and ring must use the same device");
  }
  absl::StatusOr<MetadataSlot*> slot = state_->Resolve(token_);
  if (!slot.ok()) return slot.status();
  absl::StatusOr<BufferView> source =
      (*slot)->pinned.View(ByteRange{ByteCount(0), state_->slot_bytes});
  if (!source.ok()) return source.status();
  absl::StatusOr<MutableBufferView> destination =
      (*slot)->device.MutableView(ByteRange{ByteCount(0), state_->slot_bytes});
  if (!destination.ok()) return destination.status();
  absl::Status copy = CopyAsync(CopyRequest{*source, *destination, state_->slot_bytes}, transfer,
                                *state_->api, state_->health);
  if (!copy.ok()) return copy;
  absl::StatusOr<CudaEventLease> event = state_->event_pool->Acquire();
  if (!event.ok()) return event.status();
  absl::Status record = event->Record(transfer);
  if (!record.ok()) return record;
  absl::Status wait = event->WaitOn(compute);
  if (!wait.ok()) return wait;
  (*slot)->upload_event.emplace(std::move(*event));
  (*slot)->state = MetadataState::kInFlight;
  sealed_ = true;
  return destination->AsConst();
}

absl::StatusOr<BufferView> CudaMetadataLease::device_view() const {
  if (state_ == nullptr || !sealed_) {
    return absl::FailedPreconditionError("cuda.metadata.device_view: metadata has not been sealed");
  }
  absl::StatusOr<MetadataSlot*> slot = state_->Resolve(token_);
  if (!slot.ok()) return slot.status();
  return (*slot)->device.View(ByteRange{ByteCount(0), state_->slot_bytes});
}

absl::Status CudaMetadataLease::Release() {
  if (state_ == nullptr) return StaleMetadata("release");
  absl::StatusOr<MetadataSlot*> slot = state_->Resolve(token_);
  if (!slot.ok()) return slot.status();
  if (sealed_) {
    if (!(*slot)->upload_event.has_value()) {
      return absl::InternalError("cuda.metadata.release: in-flight slot has no upload event");
    }
    absl::StatusOr<FencePoll> poll = (*slot)->upload_event->Poll();
    if (!poll.ok()) return poll.status();
    if (poll->state == FenceState::kPending) {
      return WithErrorReason(
          absl::FailedPreconditionError("cuda.metadata.release: upload remains pending"),
          ErrorReason::kPendingResource);
    }
    absl::Status event_release = (*slot)->upload_event->Release();
    if (!event_release.ok()) return event_release;
    (*slot)->upload_event.reset();
  }
  (*slot)->state = MetadataState::kFree;
  state_.reset();
  sealed_ = false;
  return absl::OkStatus();
}

absl::StatusOr<CudaMetadataRing> CudaMetadataRing::Create(
    uint32_t slots, ByteCount slot_bytes, CudaPinnedAllocator& pinned_allocator,
    CudaDeviceAllocator& device_allocator, CudaEventPool& event_pool, DeviceId device,
    const CudaApi& api, CudaHealth* health, PoolGeneration initial_generation) {
  if (slots < 2 || slot_bytes.value() == 0) {
    return absl::InvalidArgumentError(
        "cuda.metadata.geometry: at least two non-empty slots are required");
  }
  std::shared_ptr<MetadataRingState> state;
  try {
    state = std::make_shared<MetadataRingState>();
    state->slots.reserve(slots);
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("cuda.metadata: ring state allocation failed");
  }
  state->event_pool = &event_pool;
  state->api = &api;
  state->health = health;
  state->device = device;
  state->slot_bytes = slot_bytes;
  for (uint32_t index = 0; index < slots; ++index) {
    static_cast<void>(index);
    const AllocationRequest pinned_request{Device::Host(), MemoryKind::kPinnedHost, slot_bytes,
                                           ByteCount(64), MemoryCategory::kExecutionMetadata};
    const AllocationRequest device_request{Device::Cuda(device), MemoryKind::kDevice, slot_bytes,
                                           ByteCount(64), MemoryCategory::kExecutionMetadata};
    absl::StatusOr<Buffer> pinned = pinned_allocator.Allocate(pinned_request);
    if (!pinned.ok()) {
      static_cast<void>(ReleaseSlotBuffers(*state));
      return pinned.status();
    }
    absl::StatusOr<Buffer> device_buffer = device_allocator.Allocate(device_request);
    if (!device_buffer.ok()) {
      static_cast<void>(pinned->Release());
      static_cast<void>(ReleaseSlotBuffers(*state));
      return device_buffer.status();
    }
    state->slots.push_back(MetadataSlot{std::move(*pinned), std::move(*device_buffer),
                                        initial_generation, MetadataState::kFree, std::nullopt});
  }
  return CudaMetadataRing(std::move(state));
}

CudaMetadataRing::CudaMetadataRing(std::shared_ptr<MetadataRingState> state) noexcept
    : state_(std::move(state)) {}

CudaMetadataRing::~CudaMetadataRing() noexcept {
  if (state_ != nullptr) static_cast<void>(Close());
}

absl::StatusOr<CudaMetadataLease> CudaMetadataRing::Acquire() {
  if (state_ == nullptr || state_->closed) {
    return absl::FailedPreconditionError("cuda.metadata.acquire: ring is closed");
  }
  if (state_->health != nullptr) {
    absl::Status accepting = state_->health->CheckAcceptingWork();
    if (!accepting.ok()) return accepting;
  }
  for (size_t index = 0; index < state_->slots.size(); ++index) {
    MetadataSlot& slot = state_->slots[index];
    if (slot.state != MetadataState::kFree) continue;
    if (slot.generation.value() == std::numeric_limits<uint64_t>::max()) {
      slot.state = MetadataState::kRetired;
      continue;
    }
    slot.generation = PoolGeneration(slot.generation.value() + 1);
    slot.state = MetadataState::kHostWritable;
    return CudaMetadataLease(
        state_, MetadataToken{PoolSlotId(static_cast<uint32_t>(index)), slot.generation});
  }
  return WithErrorReason(absl::ResourceExhaustedError("cuda.metadata.acquire: ring exhausted"),
                         ErrorReason::kPoolExhausted);
}

absl::Status CudaMetadataRing::ValidateInvariants() const {
  if (state_ == nullptr) return absl::OkStatus();
  for (const MetadataSlot& slot : state_->slots) {
    if ((slot.state == MetadataState::kInFlight) != slot.upload_event.has_value()) {
      return WithErrorReason(absl::InternalError("cuda.metadata: slot/event invariant failed"),
                             ErrorReason::kInvariantViolation);
    }
  }
  return absl::OkStatus();
}

absl::Status CudaMetadataRing::Close() {
  if (state_ == nullptr || state_->closed) return absl::OkStatus();
  for (const MetadataSlot& slot : state_->slots) {
    if (slot.state == MetadataState::kHostWritable || slot.state == MetadataState::kInFlight) {
      return WithErrorReason(
          absl::FailedPreconditionError("cuda.metadata.close: live leases remain"),
          ErrorReason::kPendingResource);
    }
  }
  absl::Status release = ReleaseSlotBuffers(*state_);
  if (!release.ok()) return release;
  state_->closed = true;
  state_.reset();
  return absl::OkStatus();
}

}  // namespace inferx::cuda
