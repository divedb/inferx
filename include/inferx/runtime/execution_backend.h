#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/clock.h"
#include "inferx/engine/execution_completion.h"
#include "inferx/input/model_package.h"
#include "inferx/runtime/model_handle.h"
#include "inferx/scheduler/step_plan.h"
#include "inferx/tensor/device.h"

namespace inferx::runtime {

// Hardware-neutral model-load request (ADR 0035). The package was validated
// by M3; the backend plans memory, uploads weights, prepares operators, and
// warms up before publishing a ready ModelInstance.
struct ModelLoadPlan {
  const input::ValidatedModelPackage* package = nullptr;
  // Local model root; the package owns relative artifact identity only.
  std::string model_root;
  uint64_t max_prefill_tokens = 512;
  uint64_t context_capacity = 4096;
  uint64_t weight_alignment_bytes = 256;
};

struct BackendCapabilities {
  std::string_view name;
  DeviceKind kind = DeviceKind::kHost;
  // True when Submit executes synchronously and CompleteReady drains
  // immediately (the CPU reference backend); asynchronous backends queue
  // device work and complete on fences.
  bool synchronous = false;
};

// The batch-one M5 execution contract. Implementations: the CPU reference
// backend (tests/oracle) and the CUDA backend under kernels/cuda/execution
// (production). M5 accepts one ScheduledSequence per StepPlan, one loaded
// model, and one in-flight ticket.
class ExecutionBackend {
 public:
  virtual ~ExecutionBackend() = default;

  [[nodiscard]] virtual BackendCapabilities capabilities() const = 0;

  // Transactional: nothing is published on failure; repeated loads allocate
  // new slots and increment generations.
  [[nodiscard]] virtual absl::StatusOr<ModelHandle> Load(const ModelLoadPlan& plan) = 0;
  [[nodiscard]] virtual absl::Status Unload(ModelHandle handle) = 0;

  // Stop-token ids from the validated model/tokenizer metadata for a loaded
  // model; the runner turns a matching committed output into a real EOS
  // finish (m5.md section 13.4).
  [[nodiscard]] virtual std::span<const int32_t> stop_token_ids(ModelHandle handle) const = 0;

  // Validates the one-item plan, prepares the KV append, executes (or
  // queues) the semantic forward, and owns the plan lease until
  // Acknowledge. `step_tokens` are the input-token values for the plan's
  // token range; the backend copies them into ticket-owned metadata before
  // returning and never reads them afterwards.
  [[nodiscard]] virtual absl::StatusOr<ExecutionTicket> Submit(
      scheduler::StepPlanLease plan, ModelHandle model, std::span<const int32_t> step_tokens) = 0;

  // Drains completed tickets in submission order. A successful completion
  // carries exactly one output token selected by the backend's greedy rule.
  [[nodiscard]] virtual absl::StatusOr<size_t> CompleteReady(
      MonotonicTime now, std::span<ExecutionCompletion> output) = 0;

  // Releases plan, activation, metadata, logits, and KV transaction
  // ownership for a drained ticket.
  [[nodiscard]] virtual absl::Status Acknowledge(ExecutionTicketId ticket) = 0;
};

}  // namespace inferx::runtime
