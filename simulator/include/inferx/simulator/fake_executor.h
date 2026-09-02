// Deterministic ticket owner and completion producer.

#ifndef INFERX_SIMULATOR_FAKE_EXECUTOR_H_
#define INFERX_SIMULATOR_FAKE_EXECUTOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"
#include "inferx/engine/execution_completion.h"
#include "inferx/scheduler/step_plan.h"
#include "inferx/simulator/latency_model.h"

namespace inferx::simulator {

enum class FailureTargetKind : uint8_t {
  kStep,
  kRequest,
  kSubmission,
};

[[nodiscard]] absl::string_view ToString(FailureTargetKind kind);
[[nodiscard]] std::optional<FailureTargetKind> FailureTargetKindFromName(absl::string_view name);

struct FailureRule {
  FailureTargetKind target_kind = FailureTargetKind::kStep;
  uint64_t target_id = 0;
  absl::Status status;
  ErrorReason reason = ErrorReason::kExecutorFailure;
  bool persistent = false;
};

[[nodiscard]] uint64_t FakeTokenBits(RequestId request, TokenOffset position) noexcept;
[[nodiscard]] TokenId FakeToken(RequestId request, TokenOffset position) noexcept;

class FakeExecutor {
 public:
  static absl::StatusOr<std::unique_ptr<FakeExecutor>> Create(LatencyModel latency,
                                                              size_t max_tickets,
                                                              size_t max_failure_rules);

  [[nodiscard]] absl::Status AddFailureRule(FailureRule rule);
  [[nodiscard]] absl::StatusOr<ExecutionTicket> Submit(scheduler::StepPlanLease plan);
  [[nodiscard]] absl::StatusOr<size_t> CompleteReady(MonotonicTime now,
                                                     std::span<ExecutionCompletion> output);
  [[nodiscard]] absl::Status Acknowledge(ExecutionTicketId ticket);
  [[nodiscard]] std::optional<MonotonicTime> NextCompletionTime() const;
  [[nodiscard]] absl::Status Validate() const;

  [[nodiscard]] size_t live_tickets() const noexcept { return pending_.size(); }
  [[nodiscard]] size_t delivered_tickets() const noexcept;
  [[nodiscard]] size_t unconsumed_failure_rules() const noexcept;
  [[nodiscard]] uint64_t next_submission_ordinal() const noexcept {
    return next_submission_ordinal_;
  }

 private:
  struct Limits {
    size_t max_tickets = 0;
    size_t max_failure_rules = 0;
  };

  struct PendingTicket {
    ExecutionTicket ticket;
    uint64_t submission_ordinal = 0;
    MonotonicTime completion_time{};
    scheduler::StepPlanLease plan;
    bool delivered = false;
  };

  struct StoredFailureRule {
    FailureRule rule;
    bool consumed = false;
    bool matched_delivery = false;
  };

  FakeExecutor(LatencyModel latency, Limits limits);
  [[nodiscard]] bool RuleMatches(const FailureRule& rule, const PendingTicket& pending,
                                 const scheduler::ScheduledSequence& item) const noexcept;

  LatencyModel latency_;
  size_t max_tickets_ = 0;
  size_t max_failure_rules_ = 0;
  std::vector<PendingTicket> pending_;
  std::vector<StoredFailureRule> failure_rules_;
  uint64_t next_ticket_id_ = 1;
  uint64_t next_submission_ordinal_ = 1;
};

}  // namespace inferx::simulator

#endif  // INFERX_SIMULATOR_FAKE_EXECUTOR_H_
