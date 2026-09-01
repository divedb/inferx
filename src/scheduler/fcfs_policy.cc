#include "inferx/scheduler/fcfs_policy.h"

#include "absl/status/status.h"
#include "inferx/base/status.h"

namespace inferx::scheduler {
namespace {

bool IsEligibleState(RequestState state) {
  return state == RequestState::kQueued || state == RequestState::kPrefillReady ||
         state == RequestState::kDecodeReady;
}

bool OrderedBefore(const RequestSchedulingView& left, const RequestSchedulingView& right) {
  if (left.arrival_time != right.arrival_time) {
    return left.arrival_time < right.arrival_time;
  }
  if (left.arrival_ordinal != right.arrival_ordinal) {
    return left.arrival_ordinal < right.arrival_ordinal;
  }
  return left.request < right.request;
}

absl::Status PolicyError(absl::StatusCode code, absl::string_view message) {
  return WithErrorReason(absl::Status(code, message), ErrorReason::kInvariantViolation);
}

}  // namespace

absl::StatusOr<size_t> FcfsPolicy::Select(const SchedulingSnapshot& snapshot,
                                          std::span<size_t> output_indices) const {
  for (size_t index = 1; index < snapshot.requests.size(); ++index) {
    if (!OrderedBefore(snapshot.requests[index - 1], snapshot.requests[index])) {
      return PolicyError(absl::StatusCode::kFailedPrecondition,
                         "scheduler.fcfs: snapshot is not in strict arrival order");
    }
  }

  size_t count = 0;
  for (const RequestSchedulingView& request : snapshot.requests) {
    const bool expired = request.deadline.has_value() && snapshot.now >= *request.deadline;
    if (IsEligibleState(request.state) && !expired) {
      ++count;
    }
  }
  if (count > output_indices.size()) {
    return WithErrorReason(
        absl::ResourceExhaustedError("scheduler.fcfs: output scratch is too small"),
        ErrorReason::kCapacityExhausted);
  }

  size_t output = 0;
  for (size_t index = 0; index < snapshot.requests.size(); ++index) {
    const RequestSchedulingView& request = snapshot.requests[index];
    const bool expired = request.deadline.has_value() && snapshot.now >= *request.deadline;
    if (IsEligibleState(request.state) && !expired) {
      output_indices[output++] = index;
    }
  }
  return output;
}

}  // namespace inferx::scheduler
