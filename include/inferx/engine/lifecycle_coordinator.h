// Single-thread-owned orchestration of FSM, resources, and responses.

#ifndef INFERX_ENGINE_LIFECYCLE_COORDINATOR_H_
#define INFERX_ENGINE_LIFECYCLE_COORDINATOR_H_

#include <cstddef>

#include "absl/status/status.h"
#include "inferx/api/generate_request.h"
#include "inferx/api/response_event.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/engine/event_sink.h"
#include "inferx/engine/execution_completion.h"
#include "inferx/engine/request_controller.h"
#include "inferx/engine/request_registry.h"
#include "inferx/scheduler/resource_accountant.h"
#include "inferx/scheduler/step_plan.h"

namespace inferx {

class LifecycleCoordinator {
 public:
  LifecycleCoordinator(RequestRegistry& registry, scheduler::ResourceAccountant& resources,
                       ResponseSink& responses, LifecycleObserver& observer,
                       size_t max_live_requests) noexcept;

  [[nodiscard]] absl::Status Admit(GenerateRequest request, MonotonicTime arrival);
  [[nodiscard]] absl::Status Reserve(RequestId request, scheduler::ResourceCost cost,
                                     MonotonicTime now);
  [[nodiscard]] absl::Status MarkSubmitted(const scheduler::StepPlan& plan,
                                           const ExecutionTicket& ticket, MonotonicTime now);
  [[nodiscard]] absl::Status Complete(const ExecutionCompletion& completion, MonotonicTime now);
  [[nodiscard]] absl::Status Cancel(RequestId request, FinishReason finish, absl::Status status,
                                    ErrorReason reason, MonotonicTime now);
  [[nodiscard]] absl::Status Fail(RequestId request, absl::Status status, ErrorReason reason,
                                  MonotonicTime now);
  [[nodiscard]] absl::Status Preempt(RequestId request, MonotonicTime now);
  [[nodiscard]] absl::Status Requeue(RequestId request, MonotonicTime now);
  [[nodiscard]] absl::Status EmitTerminal(RequestId request, FinishReason reason,
                                          MonotonicTime now);
  // M5 schema extension: the committed output matched a stop id. Legal from
  // kPrefillReady/kDecodeReady; the terminal emission follows with
  // FinishReason::kEos (m5.md section 13.4).
  [[nodiscard]] absl::Status Stop(RequestId request, MonotonicTime now);

 private:
  [[nodiscard]] absl::Status ObserveTransition(const RequestContext& request, RequestState from,
                                               RequestEventKind event, MonotonicTime now);
  [[nodiscard]] absl::StatusOr<ResourceObservation> ReleaseReservation(
      const RequestContext& request, MonotonicTime now);

  RequestRegistry& registry_;
  scheduler::ResourceAccountant& resources_;
  ResponseSink& responses_;
  LifecycleObserver& observer_;
  size_t max_live_requests_;
  RequestController controller_;
};

}  // namespace inferx

#endif  // INFERX_ENGINE_LIFECYCLE_COORDINATOR_H_
