#include "tests/fakes/fake_fence.h"

#include <cstdint>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "inferx/base/status.h"

namespace inferx::testing {
namespace {

absl::Status StaleFence(absl::string_view operation) {
  return WithErrorReason(
      absl::FailedPreconditionError(absl::StrCat(operation, ": stale fence token")),
      ErrorReason::kStaleFence);
}

}  // namespace

FakeFenceDomain::FakeFenceDomain(uint32_t slot_count, FenceGeneration initial_generation)
    : slots_(slot_count) {
  for (Slot& slot : slots_) {
    slot.generation = initial_generation;
  }
}

absl::StatusOr<CompletionFence> FakeFenceDomain::Acquire(Device device) {
  for (size_t index = 0; index < slots_.size(); ++index) {
    Slot& slot = slots_[index];
    if (slot.active) {
      continue;
    }
    if (slot.generation.value() == std::numeric_limits<uint64_t>::max()) {
      continue;
    }
    slot.generation = FenceGeneration(slot.generation.value() + 1);
    slot.state = FenceState::kPending;
    slot.status = absl::OkStatus();
    slot.active = true;
    slot.abandoned = false;
    slot.device = device;
    return CompletionFence(
        this, FenceToken{device, FenceSlotId(static_cast<uint32_t>(index)), slot.generation});
  }
  return WithErrorReason(absl::ResourceExhaustedError("fake_fence.acquire: no free slots"),
                         ErrorReason::kPoolExhausted);
}

absl::StatusOr<FakeFenceDomain::Slot*> FakeFenceDomain::Resolve(FenceToken token) {
  if (token.slot.value() >= slots_.size()) {
    return StaleFence("fake_fence.resolve");
  }
  Slot& slot = slots_[token.slot.value()];
  if (!slot.active || slot.generation != token.generation || slot.device != token.device) {
    return StaleFence("fake_fence.resolve");
  }
  return &slot;
}

absl::Status FakeFenceDomain::Complete(FenceToken token) {
  absl::StatusOr<Slot*> slot = Resolve(token);
  if (!slot.ok()) {
    return slot.status();
  }
  (*slot)->state = FenceState::kComplete;
  (*slot)->status = absl::OkStatus();
  return absl::OkStatus();
}

absl::Status FakeFenceDomain::Fail(FenceToken token, absl::Status status) {
  if (status.ok()) {
    return absl::InvalidArgumentError("fake_fence.fail: failure status must not be OK");
  }
  absl::StatusOr<Slot*> slot = Resolve(token);
  if (!slot.ok()) {
    return slot.status();
  }
  (*slot)->state = FenceState::kFailed;
  (*slot)->status = std::move(status);
  return absl::OkStatus();
}

absl::StatusOr<FencePoll> FakeFenceDomain::Poll(FenceToken token) {
  absl::StatusOr<Slot*> slot = Resolve(token);
  if (!slot.ok()) {
    return slot.status();
  }
  return FencePoll{(*slot)->state, (*slot)->status};
}

absl::Status FakeFenceDomain::WaitUntil(FenceToken token, Deadline, FenceWaitReason) {
  absl::StatusOr<FencePoll> poll = Poll(token);
  if (!poll.ok()) {
    return poll.status();
  }
  if (poll->state == FenceState::kPending) {
    return absl::DeadlineExceededError("fake_fence.wait: deadline expired");
  }
  return poll->completion_status;
}

absl::Status FakeFenceDomain::Acknowledge(FenceToken token) {
  absl::StatusOr<Slot*> slot = Resolve(token);
  if (!slot.ok()) {
    return slot.status();
  }
  if ((*slot)->state == FenceState::kPending) {
    return WithErrorReason(
        absl::FailedPreconditionError("fake_fence.acknowledge: fence remains pending"),
        ErrorReason::kPendingResource);
  }
  (*slot)->active = false;
  (*slot)->abandoned = false;
  return absl::OkStatus();
}

void FakeFenceDomain::Abandon(FenceToken token) noexcept {
  absl::StatusOr<Slot*> slot = Resolve(token);
  if (slot.ok()) {
    (*slot)->abandoned = true;
  }
}

uint32_t FakeFenceDomain::abandoned_count() const noexcept {
  uint32_t count = 0;
  for (const Slot& slot : slots_) {
    if (slot.active && slot.abandoned) {
      ++count;
    }
  }
  return count;
}

absl::Status FakeFenceDomain::ReclaimAbandoned() {
  for (Slot& slot : slots_) {
    if (slot.active && slot.abandoned && slot.state != FenceState::kPending) {
      slot.active = false;
      slot.abandoned = false;
    }
  }
  return absl::OkStatus();
}

}  // namespace inferx::testing
