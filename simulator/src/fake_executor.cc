#include "inferx/simulator/fake_executor.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/status.h"

namespace inferx::simulator {

absl::string_view ToString(FailureTargetKind kind) {
  switch (kind) {
    case FailureTargetKind::kStep:
      return "step";
    case FailureTargetKind::kRequest:
      return "request";
    case FailureTargetKind::kSubmission:
      return "submission";
  }
  return "unknown";
}

std::optional<FailureTargetKind> FailureTargetKindFromName(absl::string_view name) {
  if (name == "step") {
    return FailureTargetKind::kStep;
  }
  if (name == "request") {
    return FailureTargetKind::kRequest;
  }
  if (name == "submission") {
    return FailureTargetKind::kSubmission;
  }
  return std::nullopt;
}

uint64_t FakeTokenBits(RequestId request, TokenOffset position) noexcept {
  uint64_t bits = request.value() + 0x9e3779b97f4a7c15ULL;
  bits += static_cast<uint64_t>(position.value()) * 0x9e3779b97f4a7c15ULL;
  bits = (bits ^ (bits >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  bits = (bits ^ (bits >> 27U)) * 0x94d049bb133111ebULL;
  return bits ^ (bits >> 31U);
}

TokenId FakeToken(RequestId request, TokenOffset position) noexcept {
  return TokenId(static_cast<int32_t>(FakeTokenBits(request, position) % 32000ULL));
}

FakeExecutor::FakeExecutor(LatencyModel latency, Limits limits)
    : latency_(latency),
      max_tickets_(limits.max_tickets),
      max_failure_rules_(limits.max_failure_rules) {
  pending_.reserve(max_tickets_);
  failure_rules_.reserve(max_failure_rules_);
}

absl::StatusOr<std::unique_ptr<FakeExecutor>> FakeExecutor::Create(LatencyModel latency,
                                                                   size_t max_tickets,
                                                                   size_t max_failure_rules) {
  if (max_tickets == 0) {
    return WithErrorReason(
        absl::InvalidArgumentError("simulator.executor: max_tickets must be positive"),
        ErrorReason::kInvalidConfig);
  }
  try {
    return std::unique_ptr<FakeExecutor>(
        new FakeExecutor(latency, Limits{max_tickets, max_failure_rules}));
  } catch (const std::bad_alloc&) {
    return WithErrorReason(
        absl::ResourceExhaustedError("simulator.executor: bounded storage allocation failed"),
        ErrorReason::kCapacityExhausted);
  } catch (...) {
    return WithErrorReason(absl::InternalError("simulator.executor: unexpected exception"),
                           ErrorReason::kInvariantViolation);
  }
}

absl::Status FakeExecutor::AddFailureRule(FailureRule rule) {
  if (rule.status.ok() || rule.reason == ErrorReason::kNone) {
    return WithErrorReason(
        absl::InvalidArgumentError("simulator.executor.failure: status and reason must be non-OK"),
        ErrorReason::kInvalidWorkload);
  }
  if (failure_rules_.size() >= max_failure_rules_) {
    return WithErrorReason(
        absl::ResourceExhaustedError("simulator.executor.failure: rule capacity exhausted"),
        ErrorReason::kCapacityExhausted);
  }
  failure_rules_.push_back(StoredFailureRule{std::move(rule), false, false});
  return absl::OkStatus();
}

absl::StatusOr<ExecutionTicket> FakeExecutor::Submit(scheduler::StepPlanLease plan) {
  if (!plan.valid() || plan.plan().sequences.empty()) {
    return WithErrorReason(
        absl::InvalidArgumentError("simulator.executor.submit: plan lease is empty"),
        ErrorReason::kExecutorRejected);
  }
  if (pending_.size() >= max_tickets_) {
    return WithErrorReason(
        absl::ResourceExhaustedError("simulator.executor.submit: ticket capacity exhausted"),
        ErrorReason::kExecutorRejected);
  }
  if (next_ticket_id_ == std::numeric_limits<uint64_t>::max() ||
      next_submission_ordinal_ == std::numeric_limits<uint64_t>::max()) {
    return WithErrorReason(
        absl::OutOfRangeError("simulator.executor.submit: identity generator exhausted"),
        ErrorReason::kInvariantViolation);
  }
  absl::StatusOr<MonotonicTime> completion_time = latency_.CompletionTime(plan.plan());
  if (!completion_time.ok()) {
    return completion_time.status();
  }
  if (plan.plan().sequences.size() > std::numeric_limits<uint32_t>::max()) {
    return WithErrorReason(
        absl::OutOfRangeError("simulator.executor.submit: item count exceeds schema"),
        ErrorReason::kExecutorRejected);
  }
  const ExecutionTicket ticket{ExecutionTicketId(next_ticket_id_), plan.plan().id,
                               static_cast<uint32_t>(plan.plan().sequences.size())};
  pending_.push_back(
      PendingTicket{ticket, next_submission_ordinal_, *completion_time, std::move(plan), false});
  ++next_ticket_id_;
  ++next_submission_ordinal_;
  return ticket;
}

absl::StatusOr<size_t> FakeExecutor::CompleteReady(MonotonicTime now,
                                                   std::span<ExecutionCompletion> output) {
  PendingTicket* selected = nullptr;
  for (PendingTicket& pending : pending_) {
    if (pending.delivered || pending.completion_time > now) {
      continue;
    }
    if (selected == nullptr || pending.completion_time < selected->completion_time ||
        (pending.completion_time == selected->completion_time &&
         pending.submission_ordinal < selected->submission_ordinal)) {
      selected = &pending;
    }
  }
  if (selected == nullptr) {
    return size_t{0};
  }
  if (output.size() < selected->ticket.item_count) {
    return WithErrorReason(
        absl::ResourceExhaustedError("simulator.executor.complete: output cannot hold next ticket"),
        ErrorReason::kCapacityExhausted);
  }
  for (StoredFailureRule& stored : failure_rules_) {
    stored.matched_delivery = false;
  }

  const scheduler::StepPlan& plan = selected->plan.plan();
  for (size_t index = 0; index < plan.sequences.size(); ++index) {
    const scheduler::ScheduledSequence& item = plan.sequences[index];
    const FailureRule* failure = nullptr;
    for (StoredFailureRule& stored : failure_rules_) {
      if (!stored.consumed && RuleMatches(stored.rule, *selected, item)) {
        stored.matched_delivery = true;
        if (failure == nullptr) {
          failure = &stored.rule;
        }
      }
    }
    const bool success = failure == nullptr;
    output[index] = ExecutionCompletion{
        .ticket = selected->ticket.id,
        .step = selected->ticket.step,
        .item_ordinal = static_cast<uint32_t>(index),
        .item_count = selected->ticket.item_count,
        .request = item.request,
        .sequence = item.sequence,
        .epoch = item.epoch,
        .kind = item.kind,
        .scheduled_tokens = item.input_tokens,
        .status = success ? absl::OkStatus() : failure->status,
        .error_reason = success ? ErrorReason::kNone : failure->reason,
        .output_token = success
                            ? std::optional<TokenId>(FakeToken(item.request, item.output_position))
                            : std::nullopt,
    };
  }
  for (StoredFailureRule& stored : failure_rules_) {
    if (stored.matched_delivery && !stored.rule.persistent) {
      stored.consumed = true;
    }
  }
  selected->delivered = true;
  return plan.sequences.size();
}

absl::Status FakeExecutor::Acknowledge(ExecutionTicketId ticket) {
  auto found =
      std::find_if(pending_.begin(), pending_.end(),
                   [ticket](const PendingTicket& pending) { return pending.ticket.id == ticket; });
  if (found == pending_.end()) {
    return WithErrorReason(
        absl::NotFoundError(absl::StrCat("simulator.executor.ticket.", ticket.value(),
                                         ": unknown or already acknowledged")),
        ErrorReason::kUnknownResource);
  }
  if (!found->delivered) {
    return WithErrorReason(
        absl::FailedPreconditionError("simulator.executor.acknowledge: ticket is still pending"),
        ErrorReason::kStaleCompletion);
  }
  pending_.erase(found);
  return absl::OkStatus();
}

std::optional<MonotonicTime> FakeExecutor::NextCompletionTime() const {
  std::optional<MonotonicTime> next;
  uint64_t next_ordinal = 0;
  for (const PendingTicket& pending : pending_) {
    if (pending.delivered) {
      continue;
    }
    if (!next.has_value() || pending.completion_time < *next ||
        (pending.completion_time == *next && pending.submission_ordinal < next_ordinal)) {
      next = pending.completion_time;
      next_ordinal = pending.submission_ordinal;
    }
  }
  return next;
}

absl::Status FakeExecutor::Validate() const {
  for (size_t left = 0; left < pending_.size(); ++left) {
    const PendingTicket& pending = pending_[left];
    if (!pending.plan.valid() || pending.plan.plan().id != pending.ticket.step ||
        pending.plan.plan().sequences.size() != pending.ticket.item_count) {
      return WithErrorReason(
          absl::InternalError("simulator.executor: ticket/plan ownership mismatch"),
          ErrorReason::kInvariantViolation);
    }
    for (size_t right = left + 1; right < pending_.size(); ++right) {
      if (pending_[right].ticket.id == pending.ticket.id ||
          pending_[right].submission_ordinal == pending.submission_ordinal) {
        return WithErrorReason(absl::InternalError("simulator.executor: duplicate ticket identity"),
                               ErrorReason::kInvariantViolation);
      }
    }
  }
  return absl::OkStatus();
}

size_t FakeExecutor::delivered_tickets() const noexcept {
  return static_cast<size_t>(
      std::count_if(pending_.begin(), pending_.end(),
                    [](const PendingTicket& pending) { return pending.delivered; }));
}

size_t FakeExecutor::unconsumed_failure_rules() const noexcept {
  return static_cast<size_t>(std::count_if(
      failure_rules_.begin(), failure_rules_.end(),
      [](const StoredFailureRule& stored) { return !stored.consumed && !stored.rule.persistent; }));
}

bool FakeExecutor::RuleMatches(const FailureRule& rule, const PendingTicket& pending,
                               const scheduler::ScheduledSequence& item) const noexcept {
  switch (rule.target_kind) {
    case FailureTargetKind::kStep:
      return rule.target_id == pending.ticket.step.value();
    case FailureTargetKind::kRequest:
      return rule.target_id == item.request.value();
    case FailureTargetKind::kSubmission:
      return rule.target_id == pending.submission_ordinal;
  }
  return false;
}

}  // namespace inferx::simulator
