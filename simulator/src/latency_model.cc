#include "inferx/simulator/latency_model.h"

#include <cstdint>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"
#include "inferx/scheduler/work_kind.h"

namespace inferx::simulator {

LatencyModel::LatencyModel(LatencyModelParameters parameters) noexcept
    : base_(parameters.base),
      prefill_per_token_(parameters.prefill_per_token),
      decode_per_sequence_(parameters.decode_per_sequence) {}

absl::StatusOr<Nanoseconds> LatencyModel::Duration(const scheduler::StepPlan& plan) const {
  if (base_ <= Nanoseconds::zero() || prefill_per_token_ < Nanoseconds::zero() ||
      decode_per_sequence_ < Nanoseconds::zero()) {
    return WithErrorReason(absl::InvalidArgumentError(
                               "simulator.latency: base must be positive and slopes nonnegative"),
                           ErrorReason::kInvalidConfig);
  }
  uint64_t prefill_tokens = 0;
  uint64_t decode_sequences = 0;
  for (const scheduler::ScheduledSequence& item : plan.sequences) {
    if (item.kind == WorkKind::kPrefill) {
      absl::StatusOr<TokenCount> count = item.input_tokens.size("simulator.latency.prefill");
      if (!count.ok()) {
        return count.status();
      }
      absl::StatusOr<uint64_t> sum =
          CheckedAdd(prefill_tokens, static_cast<uint64_t>(count->value()),
                     "simulator.latency.prefill_tokens");
      if (!sum.ok()) {
        return sum.status();
      }
      prefill_tokens = *sum;
    } else {
      absl::StatusOr<uint64_t> sum =
          CheckedAdd(decode_sequences, uint64_t{1}, "simulator.latency.decode_sequences");
      if (!sum.ok()) {
        return sum.status();
      }
      decode_sequences = *sum;
    }
  }

  absl::StatusOr<int64_t> prefill =
      CheckedMul(prefill_per_token_.count(), static_cast<int64_t>(prefill_tokens),
                 "simulator.latency.prefill");
  absl::StatusOr<int64_t> decode =
      CheckedMul(decode_per_sequence_.count(), static_cast<int64_t>(decode_sequences),
                 "simulator.latency.decode");
  if (!prefill.ok()) {
    return prefill.status();
  }
  if (!decode.ok()) {
    return decode.status();
  }
  absl::StatusOr<int64_t> subtotal =
      CheckedAdd(base_.count(), *prefill, "simulator.latency.subtotal");
  if (!subtotal.ok()) {
    return subtotal.status();
  }
  absl::StatusOr<int64_t> total = CheckedAdd(*subtotal, *decode, "simulator.latency.total");
  if (!total.ok()) {
    return total.status();
  }
  if (*total <= 0) {
    return WithErrorReason(
        absl::InvalidArgumentError("simulator.latency: total duration must be positive"),
        ErrorReason::kInvalidConfig);
  }
  return Nanoseconds(*total);
}

absl::StatusOr<MonotonicTime> LatencyModel::CompletionTime(const scheduler::StepPlan& plan) const {
  absl::StatusOr<Nanoseconds> duration = Duration(plan);
  if (!duration.ok()) {
    return duration.status();
  }
  const int64_t start = plan.planned_at.time_since_epoch().count();
  absl::StatusOr<int64_t> end =
      CheckedAdd(start, duration->count(), "simulator.latency.completion_time");
  if (!end.ok()) {
    return end.status();
  }
  return MonotonicTime(Nanoseconds(*end));
}

}  // namespace inferx::simulator
