// Conservative full-reservation admission oracle.

#ifndef INFERX_SCHEDULER_ADMISSION_CONTROLLER_H_
#define INFERX_SCHEDULER_ADMISSION_CONTROLLER_H_

#include <cstdint>

#include "absl/status/status.h"
#include "inferx/scheduler/scheduling_snapshot.h"

namespace inferx::scheduler {

enum class AdmissionKind : uint8_t {
  kReservable,
  kTemporarilyBlocked,
  kNeverFits,
};

struct AdmissionDecision {
  AdmissionKind kind = AdmissionKind::kNeverFits;
  ResourceCost cost;
  absl::Status rejection;
};

class AdmissionController {
 public:
  [[nodiscard]] AdmissionDecision Evaluate(const RequestSchedulingView& request,
                                           const ResourceSnapshot& resources) const;
};

}  // namespace inferx::scheduler

#endif  // INFERX_SCHEDULER_ADMISSION_CONTROLLER_H_
