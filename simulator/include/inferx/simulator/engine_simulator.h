// Single-thread deterministic discrete-event simulator.

#ifndef INFERX_SIMULATOR_ENGINE_SIMULATOR_H_
#define INFERX_SIMULATOR_ENGINE_SIMULATOR_H_

#include <memory>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/api/response_event.h"
#include "inferx/config/engine_config.h"
#include "inferx/simulator/replay.h"
#include "inferx/simulator/workload.h"

namespace inferx::simulator {

struct SimulationSummary {
  uint64_t events = 0;
  uint64_t steps = 0;
  uint64_t requests = 0;
  uint64_t finished = 0;
  uint64_t cancelled = 0;
  uint64_t failed = 0;
  uint64_t used_sequences = 0;
  uint64_t used_kv_tokens = 0;
  uint64_t live_tickets = 0;
  uint64_t leased_plan_slots = 0;
  bool invariants_ok = false;
};

struct SimulationResult {
  absl::Status status;
  SimulationSummary summary;
};

class EngineSimulator {
 public:
  static absl::StatusOr<std::unique_ptr<EngineSimulator>> Create(config::EngineConfig config,
                                                                 config::ModelCapabilities model,
                                                                 ReplaySink& replay);

  ~EngineSimulator();
  EngineSimulator(const EngineSimulator&) = delete;
  EngineSimulator& operator=(const EngineSimulator&) = delete;

  [[nodiscard]] absl::Status LoadWorkload(std::span<const WorkloadEvent> events);
  [[nodiscard]] absl::StatusOr<SimulationResult> Run();
  [[nodiscard]] absl::Status StepOneTimestamp();

  [[nodiscard]] std::span<const ResponseEvent> responses() const noexcept;

 private:
  struct Impl;
  explicit EngineSimulator(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::simulator

#endif  // INFERX_SIMULATOR_ENGINE_SIMULATOR_H_
