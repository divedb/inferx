#include "inferx/platform/cuda/cuda_event_pool.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "inferx/base/status.h"

namespace inferx::cuda {
namespace {

enum class EventState : uint8_t {
  kFree,
  kLeased,
  kRecorded,
  kDeferred,
  kRetired,
};

struct EventSlot {
  cudaEvent_t event = nullptr;
  FenceGeneration generation = FenceGeneration(0);
  EventState state = EventState::kFree;
  bool terminal_observed = false;
};

absl::Status StaleEvent(absl::string_view operation) {
  return WithErrorReason(
      absl::FailedPreconditionError(absl::StrCat("cuda.", operation, ": stale event token")),
      ErrorReason::kStaleFence);
}

absl::Status AnnotateCleanup(absl::Status primary, const absl::Status& cleanup) {
  if (cleanup.ok()) return primary;
  absl::Status annotated(
      primary.code(), absl::StrCat(primary.message(), "; cleanup failure: ", cleanup.ToString()));
  primary.ForEachPayload([&annotated](absl::string_view url, const absl::Cord& payload) {
    annotated.SetPayload(url, payload);
  });
  return annotated;
}

}  // namespace

struct CudaEventPool::Impl {
  const CudaApi* api = nullptr;
  CudaHealth* health = nullptr;
  DeviceId device = DeviceId(0);
  uint32_t slot_count = 0;
  bool direct_event_calls = false;
  std::vector<EventSlot> slots;
  absl::Status close_status = absl::OkStatus();
  bool closed = false;
};

CudaEventLease::CudaEventLease(CudaEventPool* pool, FenceToken token) noexcept
    : pool_(pool), token_(token) {}

CudaEventLease::~CudaEventLease() noexcept {
  if (pool_ != nullptr) {
    if (recorded_) {
      pool_->Abandon(token_);
    } else {
      static_cast<void>(pool_->ReleaseUnrecorded(token_));
    }
  }
}

CudaEventLease::CudaEventLease(CudaEventLease&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)),
      token_(other.token_),
      recorded_(std::exchange(other.recorded_, false)) {}

CudaEventLease& CudaEventLease::operator=(CudaEventLease&& other) noexcept {
  if (this != &other) {
    if (pool_ != nullptr) {
      if (recorded_)
        pool_->Abandon(token_);
      else
        static_cast<void>(pool_->ReleaseUnrecorded(token_));
    }
    pool_ = std::exchange(other.pool_, nullptr);
    token_ = other.token_;
    recorded_ = std::exchange(other.recorded_, false);
  }
  return *this;
}

absl::Status CudaEventLease::Record(CudaStream& stream) {
  if (pool_ == nullptr || recorded_) {
    return absl::FailedPreconditionError(
        "cuda.event.record: lease is inactive or already recorded");
  }
  absl::Status status = pool_->Record(token_, stream);
  if (status.ok()) recorded_ = true;
  return status;
}

absl::Status CudaEventLease::WaitOn(CudaStream& stream) {
  if (pool_ == nullptr || !recorded_) {
    return absl::FailedPreconditionError("cuda.event.wait: event has not been recorded");
  }
  return pool_->WaitOn(token_, stream);
}

absl::StatusOr<FencePoll> CudaEventLease::Poll() {
  if (pool_ == nullptr || !recorded_) {
    return absl::FailedPreconditionError("cuda.event.poll: event has not been recorded");
  }
  return pool_->Poll(token_);
}

absl::Status CudaEventLease::Release() {
  if (pool_ == nullptr) {
    return StaleEvent("event.release");
  }
  absl::Status status = recorded_ ? pool_->ReleaseLease(token_) : pool_->ReleaseUnrecorded(token_);
  if (status.ok()) {
    pool_ = nullptr;
    recorded_ = false;
  }
  return status;
}

absl::StatusOr<CompletionFence> CudaEventLease::IntoFence() {
  if (pool_ == nullptr || !recorded_) {
    return absl::FailedPreconditionError("cuda.event.into_fence: recorded event is required");
  }
  CudaEventPool* pool = std::exchange(pool_, nullptr);
  recorded_ = false;
  return CompletionFence(pool, token_);
}

CudaEventPool::CudaEventPool(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

absl::StatusOr<std::unique_ptr<CudaEventPool>> CudaEventPool::Create(
    DeviceId device, uint32_t slots, const CudaApi& api, CudaHealth* health,
    FenceGeneration initial_generation) {
  if (health != nullptr) {
    absl::Status accepting = health->CheckAcceptingWork();
    if (!accepting.ok()) return accepting;
  }
  if (slots == 0) {
    return absl::InvalidArgumentError("cuda.event_pool.slots: must be positive");
  }
  std::unique_ptr<CudaEventPool> pool;
  try {
    auto impl = std::make_unique<Impl>();
    impl->api = &api;
    impl->health = health;
    impl->device = device;
    impl->slot_count = slots;
    impl->direct_event_calls = api.event_record == &cudaEventRecord &&
                               api.event_query == &cudaEventQuery &&
                               api.stream_wait_event == &cudaStreamWaitEvent;
    impl->slots.resize(slots);
    for (EventSlot& slot : impl->slots) {
      slot.generation = initial_generation;
    }
    pool.reset(new CudaEventPool(std::move(impl)));
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("cuda.event_pool: host resource construction failed");
  }
  for (EventSlot& slot : pool->impl_->slots) {
    cudaError_t error = api.event_create_with_flags(&slot.event, cudaEventDisableTiming);
    if (error != cudaSuccess) {
      absl::Status primary = CudaErrorStatus(error, "event-create", device, health);
      return AnnotateCleanup(std::move(primary), pool->Close());
    }
  }
  return pool;
}

CudaEventPool::~CudaEventPool() noexcept {
  if (impl_ != nullptr && !impl_->closed) {
    static_cast<void>(Close());
  }
}

absl::StatusOr<CudaEventLease> CudaEventPool::Acquire() {
  if (impl_ == nullptr || impl_->closed) {
    return absl::FailedPreconditionError("cuda.event.acquire: pool is closed");
  }
  if (impl_->health != nullptr) {
    absl::Status status = impl_->health->CheckAcceptingWork();
    if (!status.ok()) return status;
  }
  for (uint32_t index = 0; index < impl_->slot_count; ++index) {
    EventSlot& slot = impl_->slots[index];
    if (slot.state != EventState::kFree) continue;
    if (slot.generation.value() == std::numeric_limits<uint64_t>::max()) {
      slot.state = EventState::kRetired;
      continue;
    }
    slot.generation = FenceGeneration(slot.generation.value() + 1);
    slot.state = EventState::kLeased;
    slot.terminal_observed = false;
    return CudaEventLease(
        this, FenceToken{Device::Cuda(impl_->device), FenceSlotId(static_cast<uint32_t>(index)),
                         slot.generation});
  }
  return WithErrorReason(absl::ResourceExhaustedError("cuda.event.acquire: event pool exhausted"),
                         ErrorReason::kPoolExhausted);
}

absl::StatusOr<cudaEvent_t> CudaEventPool::Resolve(FenceToken token, bool require_recorded) {
  if (impl_ == nullptr || token.device != Device::Cuda(impl_->device) ||
      token.slot.value() >= impl_->slot_count) {
    return StaleEvent("event.resolve");
  }
  EventSlot& slot = impl_->slots[token.slot.value()];
  if (slot.generation != token.generation || slot.state == EventState::kFree ||
      slot.state == EventState::kRetired) {
    return StaleEvent("event.resolve");
  }
  if (require_recorded && slot.state != EventState::kRecorded &&
      slot.state != EventState::kDeferred) {
    return absl::FailedPreconditionError("cuda.event.resolve: event has not been recorded");
  }
  return slot.event;
}

absl::Status CudaEventPool::Record(FenceToken token, CudaStream& stream) {
  if (impl_->health != nullptr) {
    absl::Status accepting = impl_->health->CheckAcceptingWork();
    if (!accepting.ok()) return accepting;
  }
  absl::StatusOr<cudaEvent_t> event = Resolve(token, false);
  if (!event.ok()) return event.status();
  if (stream.device() != impl_->device) {
    return absl::InvalidArgumentError("cuda.event.record: stream and event devices differ");
  }
  EventSlot& slot = impl_->slots[token.slot.value()];
  if (slot.state != EventState::kLeased) {
    return absl::FailedPreconditionError("cuda.event.record: event may be recorded only once");
  }
  cudaError_t error;
  if (impl_->direct_event_calls) [[likely]] {
    error = cudaEventRecord(*event, stream.handle());
  } else {
    error = impl_->api->event_record(*event, stream.handle());
  }
  absl::Status status = CudaErrorStatus(error, "event-record", impl_->device, impl_->health);
  if (status.ok()) slot.state = EventState::kRecorded;
  return status;
}

absl::Status CudaEventPool::WaitOn(FenceToken token, CudaStream& stream) {
  if (impl_->health != nullptr) {
    absl::Status accepting = impl_->health->CheckAcceptingWork();
    if (!accepting.ok()) return accepting;
  }
  absl::StatusOr<cudaEvent_t> event = Resolve(token, true);
  if (!event.ok()) return event.status();
  if (stream.device() != impl_->device) {
    return absl::InvalidArgumentError("cuda.event.wait: stream and event devices differ");
  }
  cudaError_t error;
  if (impl_->direct_event_calls) [[likely]] {
    error = cudaStreamWaitEvent(stream.handle(), *event, 0);
  } else {
    error = impl_->api->stream_wait_event(stream.handle(), *event, 0);
  }
  return CudaErrorStatus(error, "stream-wait-event", impl_->device, impl_->health);
}

absl::StatusOr<FencePoll> CudaEventPool::Poll(FenceToken token) {
  absl::StatusOr<cudaEvent_t> event = Resolve(token, true);
  if (!event.ok()) return event.status();
  cudaError_t error;
  if (impl_->direct_event_calls) [[likely]] {
    error = cudaEventQuery(*event);
  } else {
    error = impl_->api->event_query(*event);
  }
  if (error == cudaSuccess) {
    impl_->slots[token.slot.value()].terminal_observed = true;
    return FencePoll{FenceState::kComplete, absl::OkStatus()};
  }
  if (error == cudaErrorNotReady) {
    return FencePoll{FenceState::kPending, absl::OkStatus()};
  }
  absl::Status status = CudaErrorStatus(error, "event-query", impl_->device, impl_->health);
  impl_->slots[token.slot.value()].terminal_observed = true;
  return FencePoll{FenceState::kFailed, status};
}

absl::Status CudaEventPool::WaitUntil(FenceToken token, Deadline deadline, FenceWaitReason) {
  auto backoff = std::chrono::microseconds(50);
  while (true) {
    absl::StatusOr<FencePoll> poll = Poll(token);
    if (!poll.ok()) return poll.status();
    if (poll->state == FenceState::kComplete) return absl::OkStatus();
    if (poll->state == FenceState::kFailed) return poll->completion_status;
    const MonotonicTime now =
        std::chrono::time_point_cast<Nanoseconds>(std::chrono::steady_clock::now());
    if (now >= deadline) {
      return absl::DeadlineExceededError("cuda.event.wait: deadline expired");
    }
    std::this_thread::sleep_for(backoff);
    backoff = std::min(backoff * 2, std::chrono::microseconds(1000));
  }
}

absl::Status CudaEventPool::Acknowledge(FenceToken token) {
  absl::StatusOr<cudaEvent_t> event = Resolve(token, true);
  if (!event.ok()) return event.status();
  static_cast<void>(event);
  EventSlot& slot = impl_->slots[token.slot.value()];
  if (!slot.terminal_observed) {
    return WithErrorReason(
        absl::FailedPreconditionError(
            "cuda.event.acknowledge: terminal event state has not been observed"),
        ErrorReason::kPendingResource);
  }
  slot.state = EventState::kFree;
  slot.terminal_observed = false;
  return absl::OkStatus();
}

absl::Status CudaEventPool::ReleaseLease(FenceToken token) {
  absl::StatusOr<cudaEvent_t> event = Resolve(token, true);
  if (!event.ok()) return event.status();
  static_cast<void>(event);
  EventSlot& slot = impl_->slots[token.slot.value()];
  if (!slot.terminal_observed) {
    absl::StatusOr<FencePoll> poll = Poll(token);
    if (!poll.ok()) return poll.status();
    if (poll->state == FenceState::kPending) {
      return WithErrorReason(
          absl::FailedPreconditionError("cuda.event.release: event remains pending"),
          ErrorReason::kPendingResource);
    }
  }
  return Acknowledge(token);
}

absl::Status CudaEventPool::ReleaseUnrecorded(FenceToken token) {
  absl::StatusOr<cudaEvent_t> event = Resolve(token, false);
  if (!event.ok()) return event.status();
  EventSlot& slot = impl_->slots[token.slot.value()];
  if (slot.state != EventState::kLeased) {
    return absl::FailedPreconditionError("cuda.event.release: expected an unrecorded lease");
  }
  slot.state = EventState::kFree;
  slot.terminal_observed = false;
  return absl::OkStatus();
}

void CudaEventPool::Abandon(FenceToken token) noexcept {
  if (impl_ == nullptr || token.device != Device::Cuda(impl_->device) ||
      token.slot.value() >= impl_->slot_count) {
    return;
  }
  EventSlot& slot = impl_->slots[token.slot.value()];
  if (slot.generation != token.generation) return;
  if (slot.state == EventState::kRecorded) slot.state = EventState::kDeferred;
  if (slot.state == EventState::kLeased) slot.state = EventState::kFree;
}

absl::Status CudaEventPool::ReclaimAbandoned() {
  for (uint32_t index = 0; index < impl_->slot_count; ++index) {
    EventSlot& slot = impl_->slots[index];
    if (slot.state != EventState::kDeferred) continue;
    const FenceToken token{Device::Cuda(impl_->device), FenceSlotId(static_cast<uint32_t>(index)),
                           slot.generation};
    absl::StatusOr<FencePoll> poll = Poll(token);
    if (!poll.ok()) return poll.status();
    if (poll->state != FenceState::kPending) {
      slot.state = EventState::kFree;
      slot.terminal_observed = false;
    }
  }
  return absl::OkStatus();
}

uint32_t CudaEventPool::available_slots() const noexcept {
  uint32_t count = 0;
  if (impl_ == nullptr) return count;
  for (const EventSlot& slot : impl_->slots) {
    if (slot.state == EventState::kFree) ++count;
  }
  return count;
}

bool CudaEventPool::has_outstanding_events() const noexcept {
  if (impl_ == nullptr) return false;
  for (const EventSlot& slot : impl_->slots) {
    if (slot.state == EventState::kLeased || slot.state == EventState::kRecorded ||
        slot.state == EventState::kDeferred) {
      return true;
    }
  }
  return false;
}

absl::Status CudaEventPool::Close() {
  if (impl_ == nullptr) return absl::OkStatus();
  if (impl_->closed) return impl_->close_status;
  for (const EventSlot& slot : impl_->slots) {
    if (slot.state == EventState::kLeased || slot.state == EventState::kRecorded ||
        slot.state == EventState::kDeferred) {
      return WithErrorReason(
          absl::FailedPreconditionError("cuda.event_pool.close: live events remain"),
          ErrorReason::kPendingResource);
    }
  }
  absl::Status first = absl::OkStatus();
  for (EventSlot& slot : impl_->slots) {
    if (slot.event == nullptr) continue;
    absl::Status status = CudaErrorStatus(impl_->api->event_destroy(slot.event), "event-destroy",
                                          impl_->device, impl_->health);
    if (!status.ok() && impl_->health != nullptr) {
      impl_->health->Poison(status);
    }
    if (first.ok() && !status.ok()) first = status;
    // Destruction consumes this wrapper's ownership even when cudart reports
    // an error. Retrying an uncertain raw handle could double-destroy it.
    slot.event = nullptr;
  }
  impl_->close_status = first;
  impl_->closed = true;
  return first;
}

}  // namespace inferx::cuda
