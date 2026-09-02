// Allocation-free deterministic FCFS policy.

#ifndef INFERX_SCHEDULER_FCFS_POLICY_H_
#define INFERX_SCHEDULER_FCFS_POLICY_H_

#include <cstddef>
#include <span>

#include "absl/status/statusor.h"
#include "inferx/scheduler/scheduling_snapshot.h"

namespace inferx::scheduler {

class FcfsPolicy {
 public:
  [[nodiscard]] absl::StatusOr<size_t> Select(const SchedulingSnapshot& snapshot,
                                              std::span<size_t> output_indices) const;
};

}  // namespace inferx::scheduler

#endif  // INFERX_SCHEDULER_FCFS_POLICY_H_
