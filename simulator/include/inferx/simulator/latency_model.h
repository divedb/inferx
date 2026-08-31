// Deterministic checked fake-execution latency (m1.md section 13.2).

#ifndef INFERX_SIMULATOR_LATENCY_MODEL_H_
#define INFERX_SIMULATOR_LATENCY_MODEL_H_

#include "absl/status/statusor.h"
#include "inferx/base/clock.h"
#include "inferx/scheduler/step_plan.h"

namespace inferx::simulator {

struct LatencyModelParameters {
  Nanoseconds base{};
  Nanoseconds prefill_per_token{};
  Nanoseconds decode_per_sequence{};
};

class LatencyModel {
 public:
  explicit LatencyModel(LatencyModelParameters parameters) noexcept;

  [[nodiscard]] absl::StatusOr<Nanoseconds> Duration(const scheduler::StepPlan& plan) const;
  [[nodiscard]] absl::StatusOr<MonotonicTime> CompletionTime(const scheduler::StepPlan& plan) const;

 private:
  Nanoseconds base_;
  Nanoseconds prefill_per_token_;
  Nanoseconds decode_per_sequence_;
};

}  // namespace inferx::simulator

#endif  // INFERX_SIMULATOR_LATENCY_MODEL_H_
