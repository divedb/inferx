// Request state vocabulary (m1.md section 11.1). Shared closed enum: the
// lifecycle target owns the names only — no mutable request data, no
// dependencies on scheduler or engine types.

#ifndef INFERX_LIFECYCLE_REQUEST_STATE_H_
#define INFERX_LIFECYCLE_REQUEST_STATE_H_

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"

namespace inferx {

enum class RequestState : uint8_t {
  kReceived,
  kTokenizing,
  kQueued,
  kReserving,
  kPrefillReady,
  kPrefilling,
  kDecodeReady,
  kDecoding,
  kPreempted,
  kCancelling,
  kFinishing,
  kFinished,
  kCancelled,
  kFailed,
};

// Exact lowercase snake-case names; replay parsing accepts only these
// spellings. Total switch: adding an enum value without a case fails the
// build (and the exhaustive-switch tests).
[[nodiscard]] absl::string_view ToString(RequestState state);
[[nodiscard]] std::optional<RequestState> RequestStateFromName(absl::string_view name);

// Finished/Cancelled/Failed are terminal and absorbing.
[[nodiscard]] constexpr bool IsTerminal(RequestState state) {
  return state == RequestState::kFinished || state == RequestState::kCancelled ||
         state == RequestState::kFailed;
}

// In-flight states own submitted work (a completion must arrive or drain).
[[nodiscard]] constexpr bool IsInFlight(RequestState state) {
  return state == RequestState::kPrefilling || state == RequestState::kDecoding ||
         state == RequestState::kCancelling;
}

}  // namespace inferx

#endif  // INFERX_LIFECYCLE_REQUEST_STATE_H_
