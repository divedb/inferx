#pragma once

#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/clock.h"
#include "inferx/engine/execution_completion.h"
#include "inferx/model/forward_batch.h"
#include "inferx/runtime/execution_backend.h"
#include "inferx/runtime/model_handle.h"
#include "inferx/scheduler/step_plan.h"

namespace inferx::runtime {

// Reference ExecutionBackend over the M4 CPU kernels (m5.md section 16.1).
// It is the correctness oracle and test surface: host FP32 execution,
// synchronous submit, immediate completion. It is not a production
// generator and advertises itself as such through `capabilities()`.
class CpuExecutionBackend final : public ExecutionBackend {
 public:
  CpuExecutionBackend();
  ~CpuExecutionBackend() override;

  CpuExecutionBackend(const CpuExecutionBackend&) = delete;
  CpuExecutionBackend& operator=(const CpuExecutionBackend&) = delete;

  [[nodiscard]] BackendCapabilities capabilities() const override;
  [[nodiscard]] absl::StatusOr<ModelHandle> Load(const ModelLoadPlan& plan) override;
  [[nodiscard]] absl::Status Unload(ModelHandle handle) override;
  [[nodiscard]] std::span<const int32_t> stop_token_ids(ModelHandle handle) const override;
  [[nodiscard]] absl::StatusOr<ExecutionTicket> Submit(
      scheduler::StepPlanLease plan, ModelHandle model,
      std::span<const int32_t> step_tokens) override;
  [[nodiscard]] absl::StatusOr<size_t> CompleteReady(
      MonotonicTime now, std::span<ExecutionCompletion> output) override;
  [[nodiscard]] absl::Status Acknowledge(ExecutionTicketId ticket) override;

 private:
  struct Instance;
  struct Impl;

  // Per-step activation views at the planned offsets; logits points at the
  // instance's final-row buffer.
  [[nodiscard]] static absl::Status BuildActivationBuffers(Instance& instance, uint64_t tokens,
                                                           model::ActivationBuffers& buffers);

  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::runtime
