#include "inferx/runtime/completion_fence.h"

#include <utility>

#include "absl/status/status.h"

namespace inferx {

CompletionFence::CompletionFence(FenceDomain* domain, FenceToken token) noexcept
    : domain_(domain), token_(token) {}

CompletionFence::~CompletionFence() noexcept { Abandon(); }

CompletionFence::CompletionFence(CompletionFence&& other) noexcept
    : domain_(std::exchange(other.domain_, nullptr)),
      token_(other.token_),
      terminal_observed_(std::exchange(other.terminal_observed_, false)) {}

CompletionFence& CompletionFence::operator=(CompletionFence&& other) noexcept {
  if (this != &other) {
    Abandon();
    domain_ = std::exchange(other.domain_, nullptr);
    token_ = other.token_;
    terminal_observed_ = std::exchange(other.terminal_observed_, false);
  }
  return *this;
}

absl::StatusOr<FencePoll> CompletionFence::Poll() {
  if (domain_ == nullptr) {
    return absl::FailedPreconditionError("completion_fence.poll: fence is inactive");
  }
  absl::StatusOr<FencePoll> poll = domain_->Poll(token_);
  if (!poll.ok()) {
    return poll.status();
  }
  if (poll->state == FenceState::kComplete && !poll->completion_status.ok()) {
    return absl::InternalError("completion_fence.poll: complete state carries failure status");
  }
  if (poll->state == FenceState::kFailed && poll->completion_status.ok()) {
    return absl::InternalError("completion_fence.poll: failed state carries OK status");
  }
  if (poll->state != FenceState::kPending) {
    terminal_observed_ = true;
  }
  return *poll;
}

absl::Status CompletionFence::WaitUntil(Deadline deadline, FenceWaitReason reason) {
  if (domain_ == nullptr) {
    return absl::FailedPreconditionError("completion_fence.wait: fence is inactive");
  }
  absl::Status wait_status = domain_->WaitUntil(token_, deadline, reason);
  absl::StatusOr<FencePoll> poll = domain_->Poll(token_);
  if (poll.ok() && poll->state != FenceState::kPending) {
    terminal_observed_ = true;
  }
  return wait_status;
}

absl::Status CompletionFence::Acknowledge() {
  if (domain_ == nullptr) {
    return absl::FailedPreconditionError("completion_fence.acknowledge: fence is inactive");
  }
  if (!terminal_observed_) {
    return absl::FailedPreconditionError(
        "completion_fence.acknowledge: terminal state has not been observed");
  }
  absl::Status status = domain_->Acknowledge(token_);
  if (status.ok()) {
    domain_ = nullptr;
    terminal_observed_ = false;
  }
  return status;
}

void CompletionFence::Abandon() noexcept {
  if (domain_ != nullptr) {
    domain_->Abandon(token_);
    domain_ = nullptr;
    terminal_observed_ = false;
  }
}

}  // namespace inferx
