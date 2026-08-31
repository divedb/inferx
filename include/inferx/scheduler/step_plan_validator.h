// Cross-checks a plan against the immutable snapshot that produced it.

#ifndef INFERX_SCHEDULER_STEP_PLAN_VALIDATOR_H_
#define INFERX_SCHEDULER_STEP_PLAN_VALIDATOR_H_

#include "absl/status/status.h"
#include "inferx/scheduler/scheduling_snapshot.h"
#include "inferx/scheduler/step_plan.h"

namespace inferx::scheduler {

class StepPlanValidator {
 public:
  [[nodiscard]] absl::Status Validate(const StepPlan& plan,
                                      const SchedulingSnapshot& snapshot) const;
};

}  // namespace inferx::scheduler

#endif  // INFERX_SCHEDULER_STEP_PLAN_VALIDATOR_H_
