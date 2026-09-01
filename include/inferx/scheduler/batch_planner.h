// Allocation-free construction into a leased StepPlanBuffer.

#ifndef INFERX_SCHEDULER_BATCH_PLANNER_H_
#define INFERX_SCHEDULER_BATCH_PLANNER_H_

#include <cstddef>
#include <span>

#include "absl/status/status.h"
#include "inferx/scheduler/scheduling_snapshot.h"
#include "inferx/scheduler/step_plan.h"

namespace inferx::scheduler {

class BatchPlanner {
 public:
  [[nodiscard]] absl::Status Build(const SchedulingSnapshot& snapshot,
                                   std::span<const size_t> selected_indices,
                                   StepPlanLease& output) const;
};

}  // namespace inferx::scheduler

#endif  // INFERX_SCHEDULER_BATCH_PLANNER_H_
