#include "inferx/api/response_event.h"

namespace inferx {

absl::string_view ToString(FinishReason reason) {
  switch (reason) {
    case FinishReason::kLength:
      return "length";
    case FinishReason::kSimulatedEos:
      return "simulated_eos";
    case FinishReason::kCancelled:
      return "cancelled";
    case FinishReason::kDeadline:
      return "deadline";
    case FinishReason::kExecutorError:
      return "executor_error";
    case FinishReason::kShutdown:
      return "shutdown";
  }
  return "unknown";
}

std::optional<FinishReason> FinishReasonFromName(absl::string_view name) {
  for (uint32_t value = 0; value <= static_cast<uint32_t>(FinishReason::kShutdown); ++value) {
    const auto reason = static_cast<FinishReason>(value);
    if (ToString(reason) == name) {
      return reason;
    }
  }
  return std::nullopt;
}

}  // namespace inferx
