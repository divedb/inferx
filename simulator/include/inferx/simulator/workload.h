// Strict workload schema-v1 values and JSON Lines reader.

#ifndef INFERX_SIMULATOR_WORKLOAD_H_
#define INFERX_SIMULATOR_WORKLOAD_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "inferx/api/generate_request.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/simulator/fake_executor.h"

namespace inferx::simulator {

enum class WorkloadEventType : uint8_t {
  kSubmit,
  kCancel,
  kInjectExecutorFailure,
  kPreempt,
  kRequeue,
  kShutdown,
};

enum class ShutdownMode : uint8_t {
  kCancel,
  kDrain,
};

[[nodiscard]] absl::string_view ToString(WorkloadEventType type);
[[nodiscard]] absl::string_view ToString(ShutdownMode mode);

struct SubmitWorkloadEvent {
  GenerateRequest request;
  std::optional<TokenCount> finish_after_tokens;
};

struct CancelWorkloadEvent {
  RequestId request{0};
};

struct InjectExecutorFailureEvent {
  FailureRule rule;
};

struct PreemptWorkloadEvent {
  RequestId request{0};
};

struct RequeueWorkloadEvent {
  RequestId request{0};
};

struct ShutdownWorkloadEvent {
  ShutdownMode mode = ShutdownMode::kCancel;
};

using WorkloadPayload =
    std::variant<SubmitWorkloadEvent, CancelWorkloadEvent, InjectExecutorFailureEvent,
                 PreemptWorkloadEvent, RequeueWorkloadEvent, ShutdownWorkloadEvent>;

struct WorkloadEvent {
  uint32_t schema_version = 1;
  WorkloadEventType type = WorkloadEventType::kSubmit;
  MonotonicTime at{};
  uint64_t file_ordinal = 0;
  WorkloadPayload payload;
};

[[nodiscard]] absl::StatusOr<std::vector<WorkloadEvent>> ParseWorkloadJsonLines(
    absl::string_view text, bool require_sorted = true);
[[nodiscard]] absl::StatusOr<std::vector<WorkloadEvent>> ReadWorkloadFile(
    const std::string& path, bool require_sorted = true);
[[nodiscard]] std::string CanonicalWorkloadEvent(const WorkloadEvent& event);

}  // namespace inferx::simulator

#endif  // INFERX_SIMULATOR_WORKLOAD_H_
