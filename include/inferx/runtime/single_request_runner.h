#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/api/generate_request.h"
#include "inferx/api/response_event.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/engine/event_sink.h"
#include "inferx/engine/lifecycle_coordinator.h"
#include "inferx/engine/request_registry.h"
#include "inferx/runtime/execution_backend.h"
#include "inferx/runtime/model_handle.h"
#include "inferx/scheduler/batch_planner.h"
#include "inferx/scheduler/resource_accountant.h"
#include "inferx/scheduler/step_plan.h"
#include "inferx/scheduler/step_plan_validator.h"

namespace inferx::runtime {

// Single-request generation driver (m5.md section 13). An integration
// adapter, not a second request state machine: it drives the existing M1
// lifecycle through Admit/Reserve/plan/submit/complete/terminal and applies
// every completion through the controller. One model, one request, one
// in-flight ticket.
class SingleRequestRunner {
 public:
  struct Result {
    absl::Status status;
    FinishReason finish = FinishReason::kLength;
    std::vector<TokenId> output_tokens;
    TokenCount prompt_tokens{0};
  };

  // `sink` receives the ordered TokenDelta/TerminalResponse events; it must
  // outlive the runner.
  static absl::StatusOr<std::unique_ptr<SingleRequestRunner>> Create(ExecutionBackend& backend,
                                                                     ModelHandle model,
                                                                     uint64_t context_capacity,
                                                                     uint64_t max_prefill_tokens,
                                                                     ResponseSink& sink);

  ~SingleRequestRunner();

  SingleRequestRunner(const SingleRequestRunner&) = delete;
  SingleRequestRunner& operator=(const SingleRequestRunner&) = delete;

  // Drives one request from admission to exactly one terminal response.
  [[nodiscard]] absl::StatusOr<Result> Run(GenerateRequest request, MonotonicTime now);

 private:
  SingleRequestRunner(ExecutionBackend& backend, ModelHandle model, uint64_t max_prefill_tokens,
                      std::unique_ptr<scheduler::ResourceAccountant> resources,
                      std::unique_ptr<RequestRegistry> registry,
                      std::unique_ptr<LifecycleCoordinator> coordinator,
                      std::shared_ptr<scheduler::StepPlanPool> plan_pool);

  [[nodiscard]] absl::Status PlanSubmitComplete(RequestId request, MonotonicTime now);
  [[nodiscard]] absl::Status FinishIfTerminal(RequestId request, MonotonicTime now, Result& result);

  ExecutionBackend& backend_;
  ModelHandle model_;
  uint64_t max_prefill_tokens_;
  std::unique_ptr<scheduler::ResourceAccountant> resources_;
  std::unique_ptr<RequestRegistry> registry_;
  std::unique_ptr<LifecycleCoordinator> coordinator_;
  std::shared_ptr<scheduler::StepPlanPool> plan_pool_;
  scheduler::BatchPlanner planner_;
  scheduler::StepPlanValidator validator_;
  StepId next_step_{1};
};

}  // namespace inferx::runtime
