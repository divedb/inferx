#include "inferx/scheduler/batch_planner.h"

#include <cstdint>
#include <limits>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"

namespace inferx::scheduler {
namespace {

absl::Status PlanError(absl::string_view message) {
  return WithErrorReason(absl::InvalidArgumentError(absl::StrCat("scheduler.plan: ", message)),
                         ErrorReason::kInvariantViolation);
}

}  // namespace

absl::Status BatchPlanner::Build(const SchedulingSnapshot& snapshot,
                                 std::span<const size_t> selected_indices,
                                 StepPlanLease& output) const {
  if (!output.valid()) {
    return PlanError("output lease is empty");
  }
  internal::StepPlanBuffer& buffer = output.mutable_buffer();
  StepPlan& plan = buffer.plan;
  if (plan.id != snapshot.proposed_step || plan.planned_at != snapshot.now) {
    return PlanError("lease metadata does not match snapshot");
  }
  buffer.sequences.clear();
  plan.sequences = std::span<const ScheduledSequence>();
  plan.resources = StepResourceUse{};

  uint32_t used_tokens = 0;
  size_t previous_index = 0;
  bool has_previous = false;
  for (const size_t index : selected_indices) {
    if (index >= snapshot.requests.size()) {
      return PlanError("selected request index is out of range");
    }
    if (has_previous && index <= previous_index) {
      return PlanError("selected request indices are not unique FCFS order");
    }
    has_previous = true;
    previous_index = index;

    const RequestSchedulingView& request = snapshot.requests[index];
    if ((request.state != RequestState::kPrefillReady &&
         request.state != RequestState::kDecodeReady) ||
        !request.reservation.has_value()) {
      return PlanError("selected request is not ready with a reservation");
    }
    if (request.model != plan.model) {
      return PlanError("mixed-model plan is not supported");
    }

    const WorkKind kind =
        request.state == RequestState::kPrefillReady ? WorkKind::kPrefill : WorkKind::kDecode;
    uint32_t cost = 0;
    TokenRange input{TokenOffset(0), TokenOffset(0)};
    TokenRange read{TokenOffset(0), TokenOffset(0)};
    TokenRange write{TokenOffset(0), TokenOffset(0)};
    TokenOffset output_position(0);
    if (kind == WorkKind::kPrefill) {
      if (request.committed_output_tokens.value() != 0) {
        return PlanError("prefill request already has committed output");
      }
      cost = request.prompt_tokens.value();
      input = TokenRange{TokenOffset(0), TokenOffset(cost)};
      read = TokenRange{TokenOffset(0), TokenOffset(0)};
      write = input;
      output_position = TokenOffset(0);
    } else {
      if (request.committed_output_tokens.value() == 0) {
        return PlanError("decode request has no prior output token");
      }
      absl::StatusOr<uint32_t> end =
          CheckedAdd(request.prompt_tokens.value(), request.committed_output_tokens.value(),
                     "scheduler.plan.decode_position");
      if (!end.ok() || *end == 0) {
        return end.ok() ? PlanError("decode position is zero") : end.status();
      }
      const uint32_t begin = *end - 1;
      cost = 1;
      input = TokenRange{TokenOffset(begin), TokenOffset(*end)};
      read = TokenRange{TokenOffset(0), TokenOffset(begin)};
      write = input;
      output_position = TokenOffset(request.committed_output_tokens.value());
    }

    if (buffer.sequences.size() >= snapshot.limits.max_sequences_per_step.value()) {
      break;
    }
    absl::StatusOr<uint32_t> next_tokens = CheckedAdd(used_tokens, cost, "scheduler.plan.tokens");
    if (!next_tokens.ok()) {
      return next_tokens.status();
    }
    if (*next_tokens > snapshot.limits.max_scheduled_tokens_per_step.value()) {
      break;
    }
    if (buffer.sequences.size() >= buffer.sequences.capacity()) {
      return PlanError("leased buffer capacity is smaller than configured limit");
    }

    buffer.sequences.push_back(ScheduledSequence{
        .request = request.request,
        .sequence = request.sequence,
        .epoch = request.epoch,
        .kind = kind,
        .output_position = output_position,
        .input_tokens = input,
        .kv_read = KvReadPlan{request.sequence, read},
        .kv_write = KvWritePlan{*request.reservation, write},
    });
    used_tokens = *next_tokens;
  }

  if (buffer.sequences.size() > std::numeric_limits<uint32_t>::max()) {
    return PlanError("sequence count exceeds schema range");
  }
  plan.sequences = std::span<const ScheduledSequence>(buffer.sequences);
  plan.resources = StepResourceUse{SequenceCount(static_cast<uint32_t>(buffer.sequences.size())),
                                   TokenCount(used_tokens)};
  return absl::OkStatus();
}

}  // namespace inferx::scheduler
