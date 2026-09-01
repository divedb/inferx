#include "inferx/lifecycle/request_state.h"

namespace inferx {

absl::string_view ToString(RequestState state) {
  switch (state) {
    case RequestState::kReceived:
      return "received";
    case RequestState::kTokenizing:
      return "tokenizing";
    case RequestState::kQueued:
      return "queued";
    case RequestState::kReserving:
      return "reserving";
    case RequestState::kPrefillReady:
      return "prefill_ready";
    case RequestState::kPrefilling:
      return "prefilling";
    case RequestState::kDecodeReady:
      return "decode_ready";
    case RequestState::kDecoding:
      return "decoding";
    case RequestState::kPreempted:
      return "preempted";
    case RequestState::kCancelling:
      return "cancelling";
    case RequestState::kFinishing:
      return "finishing";
    case RequestState::kFinished:
      return "finished";
    case RequestState::kCancelled:
      return "cancelled";
    case RequestState::kFailed:
      return "failed";
  }
  return "unknown";
}

std::optional<RequestState> RequestStateFromName(absl::string_view name) {
  for (uint32_t value = 0; value <= static_cast<uint32_t>(RequestState::kFailed); ++value) {
    const auto state = static_cast<RequestState>(value);
    if (ToString(state) == name) {
      return state;
    }
  }
  return std::nullopt;
}

}  // namespace inferx
