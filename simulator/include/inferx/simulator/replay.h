// Canonical JSON Lines replay schema-v1 writer, reader, and sinks.

#ifndef INFERX_SIMULATOR_REPLAY_H_
#define INFERX_SIMULATOR_REPLAY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "inferx/api/response_event.h"
#include "inferx/base/clock.h"
#include "inferx/base/status.h"
#include "inferx/config/config_loader.h"
#include "inferx/config/engine_config.h"
#include "inferx/engine/event_sink.h"
#include "inferx/engine/execution_completion.h"
#include "inferx/scheduler/step_plan.h"
#include "inferx/simulator/workload.h"

namespace inferx::simulator {

class ReplaySink {
 public:
  virtual ~ReplaySink() = default;
  [[nodiscard]] virtual absl::Status WriteLine(absl::string_view line) = 0;
  [[nodiscard]] virtual absl::Status Close() = 0;
};

struct ReplayBufferLimits {
  size_t max_bytes = 0;
  size_t max_records = 0;
};

class InMemoryReplaySink final : public ReplaySink {
 public:
  explicit InMemoryReplaySink(ReplayBufferLimits limits);
  absl::Status WriteLine(absl::string_view line) override;
  absl::Status Close() override;
  [[nodiscard]] std::string contents() const;
  [[nodiscard]] size_t records() const noexcept { return lines_.size(); }

 private:
  size_t max_bytes_;
  size_t max_records_;
  size_t bytes_ = 0;
  bool closed_ = false;
  std::vector<std::string> lines_;
};

class FileReplaySink final : public ReplaySink {
 public:
  static absl::StatusOr<std::unique_ptr<FileReplaySink>> Create(const std::string& path,
                                                                bool overwrite);
  ~FileReplaySink() override;
  absl::Status WriteLine(absl::string_view line) override;
  absl::Status Close() override;

 private:
  struct Impl;
  explicit FileReplaySink(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

enum class IdleReason : uint8_t {
  kNoRequests,
  kWaitingForEvent,
  kWaitingForCompletion,
  kCapacityBlocked,
  kNoPlanBuffer,
};

[[nodiscard]] absl::string_view ToString(IdleReason reason);

struct ReplayFooter {
  absl::StatusCode final_status = absl::StatusCode::kOk;
  ErrorReason error_reason = ErrorReason::kNone;
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
  bool invariants_ok = true;
};

class ReplayWriter final : public LifecycleObserver {
 public:
  ReplayWriter(ReplaySink& sink, std::string canonical_config, config::ModelCapabilities model,
               size_t workload_event_count) noexcept;

  [[nodiscard]] absl::Status WriteHeader();
  [[nodiscard]] absl::Status WriteWorkload(const WorkloadEvent& event);
  [[nodiscard]] absl::Status WritePlan(const scheduler::StepPlan& plan);
  [[nodiscard]] absl::Status WriteCompletion(const ExecutionCompletion& completion,
                                             MonotonicTime now);
  [[nodiscard]] absl::Status WriteResponse(const ResponseEvent& response, MonotonicTime now);
  [[nodiscard]] absl::Status WriteIdle(IdleReason reason, MonotonicTime now,
                                       std::optional<MonotonicTime> next);
  [[nodiscard]] absl::Status WriteFooter(const ReplayFooter& footer);

  absl::Status ObserveTransition(const TransitionObservation& observation) override;
  absl::Status ObserveResource(const ResourceObservation& observation) override;

  [[nodiscard]] uint64_t next_ordinal() const noexcept { return next_ordinal_; }

 private:
  [[nodiscard]] absl::Status Write(const std::string& line);
  [[nodiscard]] std::string Prefix(absl::string_view record_type) const;

  ReplaySink& sink_;
  std::string canonical_config_;
  config::ModelCapabilities model_;
  size_t workload_event_count_;
  uint64_t next_ordinal_ = 0;
  bool header_written_ = false;
  bool footer_written_ = false;
};

struct ReplayInputs {
  config::FieldValues config_values;
  std::vector<WorkloadEvent> workload;
  std::string original_bytes;
};

[[nodiscard]] absl::Status CheckReplayJsonLines(absl::string_view text);
[[nodiscard]] absl::StatusOr<ReplayInputs> ReadReplayInputs(const std::string& path);
[[nodiscard]] absl::Status CompareReplayBytes(absl::string_view expected, absl::string_view actual);

}  // namespace inferx::simulator

#endif  // INFERX_SIMULATOR_REPLAY_H_
