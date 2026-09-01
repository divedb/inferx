#include "inferx/scheduler/step_plan_validator.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"

namespace inferx::scheduler {
namespace {

absl::Status InvalidPlan(absl::string_view detail) {
  return WithErrorReason(absl::InvalidArgumentError(absl::StrCat("scheduler.plan: ", detail)),
                         ErrorReason::kInvariantViolation);
}

bool SameRange(const TokenRange& left, const TokenRange& right) {
  return left.begin == right.begin && left.end == right.end;
}

}  // namespace

absl::Status StepPlanValidator::Validate(const StepPlan& plan,
                                         const SchedulingSnapshot& snapshot) const {
  if (plan.schema_version != 1 || plan.id.value() == 0 || plan.id != snapshot.proposed_step ||
      plan.planned_at != snapshot.now || plan.buffer_generation.value() == 0) {
    return InvalidPlan("schema, step, timestamp, or buffer generation is invalid");
  }
  if (plan.sequences.empty()) {
    return InvalidPlan("empty plans must not be submitted");
  }
  if (plan.sequences.size() != plan.resources.num_sequences.value() ||
      plan.sequences.size() > snapshot.limits.max_sequences_per_step.value()) {
    return InvalidPlan("sequence resource total does not match plan items");
  }

  uint32_t total_tokens = 0;
  size_t search_begin = 0;
  for (const ScheduledSequence& item : plan.sequences) {
    size_t matched_index = snapshot.requests.size();
    for (size_t index = search_begin; index < snapshot.requests.size(); ++index) {
      if (snapshot.requests[index].request == item.request) {
        matched_index = index;
        break;
      }
    }
    if (matched_index == snapshot.requests.size()) {
      return InvalidPlan("item is absent or out of FCFS order in snapshot");
    }
    search_begin = matched_index + 1;
    const RequestSchedulingView& request = snapshot.requests[matched_index];
    if (request.sequence != item.sequence || request.epoch != item.epoch ||
        request.model != plan.model || !request.reservation.has_value() ||
        *request.reservation != item.kv_write.reservation ||
        item.kv_read.sequence != item.sequence) {
      return InvalidPlan("item identity, model, epoch, or reservation mismatch");
    }

    TokenRange expected_input{TokenOffset(0), TokenOffset(0)};
    TokenRange expected_read{TokenOffset(0), TokenOffset(0)};
    TokenRange expected_write{TokenOffset(0), TokenOffset(0)};
    uint32_t expected_cost = 0;
    if (item.kind == WorkKind::kPrefill) {
      if (request.state != RequestState::kPrefillReady ||
          request.committed_output_tokens.value() != 0) {
        return InvalidPlan("prefill item does not match request state");
      }
      expected_cost = request.prompt_tokens.value();
      expected_input = TokenRange{TokenOffset(0), TokenOffset(expected_cost)};
      expected_read = TokenRange{TokenOffset(0), TokenOffset(0)};
      expected_write = expected_input;
      if (item.output_position != TokenOffset(0)) {
        return InvalidPlan("prefill output position is not zero");
      }
    } else {
      if (request.state != RequestState::kDecodeReady ||
          request.committed_output_tokens.value() == 0) {
        return InvalidPlan("decode item does not match request state");
      }
      absl::StatusOr<uint32_t> end =
          CheckedAdd(request.prompt_tokens.value(), request.committed_output_tokens.value(),
                     "scheduler.plan.validate.decode_position");
      if (!end.ok() || *end == 0) {
        return InvalidPlan("decode logical position overflow");
      }
      const uint32_t begin = *end - 1;
      expected_cost = 1;
      expected_input = TokenRange{TokenOffset(begin), TokenOffset(*end)};
      expected_read = TokenRange{TokenOffset(0), TokenOffset(begin)};
      expected_write = expected_input;
      if (item.output_position.value() != request.committed_output_tokens.value()) {
        return InvalidPlan("decode output position does not match committed count");
      }
    }
    if (!SameRange(item.input_tokens, expected_input) ||
        !SameRange(item.kv_read.logical_tokens, expected_read) ||
        !SameRange(item.kv_write.logical_tokens, expected_write)) {
      return InvalidPlan("logical token ranges do not match schema");
    }
    absl::StatusOr<uint32_t> next_total =
        CheckedAdd(total_tokens, expected_cost, "scheduler.plan.validate.tokens");
    if (!next_total.ok()) {
      return InvalidPlan("aggregate token cost overflow");
    }
    total_tokens = *next_total;
  }
  if (total_tokens != plan.resources.num_tokens.value() ||
      total_tokens > snapshot.limits.max_scheduled_tokens_per_step.value()) {
    return InvalidPlan("token resource total does not match plan items or budget");
  }
  return absl::OkStatus();
}

}  // namespace inferx::scheduler
