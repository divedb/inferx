// Cross-component simulator invariants (m1.md section 16).

#ifndef INFERX_SIMULATOR_INVARIANT_CHECKER_H_
#define INFERX_SIMULATOR_INVARIANT_CHECKER_H_

#include <vector>

#include "absl/status/status.h"
#include "inferx/api/response_event.h"
#include "inferx/base/token.h"
#include "inferx/engine/request_registry.h"
#include "inferx/scheduler/resource_accountant.h"
#include "inferx/scheduler/step_plan.h"
#include "inferx/simulator/fake_executor.h"

namespace inferx::simulator {

class InvariantChecker {
 public:
  InvariantChecker(const RequestRegistry& registry, const scheduler::ResourceAccountant& resources,
                   const scheduler::StepPlanPool& plan_pool, const FakeExecutor& executor,
                   const std::vector<ResponseEvent>& responses, TokenCount model_context) noexcept;

  [[nodiscard]] absl::Status Validate() const;
  [[nodiscard]] absl::Status ValidateFinal() const;

 private:
  const RequestRegistry& registry_;
  const scheduler::ResourceAccountant& resources_;
  const scheduler::StepPlanPool& plan_pool_;
  const FakeExecutor& executor_;
  const std::vector<ResponseEvent>& responses_;
  TokenCount model_context_;
};

}  // namespace inferx::simulator

#endif  // INFERX_SIMULATOR_INVARIANT_CHECKER_H_
