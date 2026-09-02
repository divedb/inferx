// Immutable scheduler input values.

#ifndef INFERX_SCHEDULER_SCHEDULING_SNAPSHOT_H_
#define INFERX_SCHEDULER_SCHEDULING_SNAPSHOT_H_

#include <cstdint>
#include <optional>
#include <span>

#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/scheduler/resource_accountant.h"

namespace inferx::scheduler {

struct RequestSchedulingView {
  RequestId request{0};
  SequenceId sequence{0};
  RequestEpoch epoch{0};
  ModelId model{0};
  RequestState state = RequestState::kReceived;
  uint64_t arrival_ordinal = 0;
  MonotonicTime arrival_time{};
  TokenCount prompt_tokens{0};
  TokenCount computed_tokens{0};
  TokenCount committed_output_tokens{0};
  TokenCount max_output_tokens{0};
  std::optional<MonotonicTime> deadline;
  std::optional<ReservationId> reservation;
};

struct SchedulingLimits {
  SequenceCount max_sequences_per_step{0};
  TokenCount max_scheduled_tokens_per_step{0};
};

struct SchedulingSnapshot {
  MonotonicTime now{};
  StepId proposed_step{0};
  std::span<const RequestSchedulingView> requests;
  SchedulingLimits limits;
  ResourceSnapshot resources;
};

}  // namespace inferx::scheduler

#endif  // INFERX_SCHEDULER_SCHEDULING_SNAPSHOT_H_
