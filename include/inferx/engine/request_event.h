// Request lifecycle events: one closed kind enum plus
// typed payload structs in a std::variant. Marker events use empty structs;
// kind/payload mismatches cannot be constructed (KindOf is the one visitor).

#ifndef INFERX_ENGINE_REQUEST_EVENT_H_
#define INFERX_ENGINE_REQUEST_EVENT_H_

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "inferx/api/response_event.h"
#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"
#include "inferx/engine/execution_completion.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/scheduler/resource_accountant.h"
#include "inferx/scheduler/work_kind.h"

namespace inferx {

enum class RequestEventKind : uint8_t {
  kStartTokenization,
  kInputReady,
  kTokenizationSucceeded,
  kTokenizationFailed,
  kBeginReservation,
  kReservationGranted,
  kReservationDeferred,
  kSubmitPrefill,
  kPrefillCompleted,
  kSubmitDecode,
  kDecodeCompleted,
  kPreempt,
  kRequeue,
  kCancelRequested,
  kDeadlineExpired,
  kInFlightDrained,
  kExecutionFailed,
  kFatalError,
  kTerminalEmitted,
};

[[nodiscard]] absl::string_view ToString(RequestEventKind kind);
[[nodiscard]] std::optional<RequestEventKind> RequestEventKindFromName(absl::string_view name);

// --- Typed payloads ---------------------------------------------------------

struct InputReadyPayload {
  std::vector<TokenId> tokens;
};

struct TokenizationSucceededPayload {
  std::vector<TokenId> tokens;
};

struct TokenizationFailedPayload {
  absl::Status failure;
};

struct ReservationGrantedPayload {
  ReservationId reservation;
  scheduler::ResourceCost cost;
};

struct ReservationDeferredPayload {};

struct SubmitPayload {
  StepId step;
  ExecutionTicketId ticket;
  RequestEpoch epoch;
  WorkKind work = WorkKind::kPrefill;
  TokenRange scheduled_range{TokenOffset(0), TokenOffset(0)};
  uint32_t item_count = 1;
};

// Distinct types per completion-bearing event so variant access is never
// ambiguous and the kind/payload pairing is total.
struct PrefillCompletedPayload {
  ExecutionCompletion completion;
};

struct DecodeCompletedPayload {
  ExecutionCompletion completion;
};

struct ExecutionFailedPayload {
  ExecutionCompletion completion;
};

// Terminal cause shared by cancel/deadline/fatal events.
struct TerminalOutcomePayload {
  absl::Status status;
  std::optional<ErrorReason> reason;
  FinishReason finish = FinishReason::kCancelled;
};

// Marker payloads.
struct StartTokenizationPayload {};
struct BeginReservationPayload {};
struct PreemptPayload {};
struct RequeuePayload {};
struct InFlightDrainedPayload {
  ExecutionCompletion completion;
};
// The successful outcome was selected when the request entered Finishing;
// this event carries the chosen success reason for the terminal record.
struct TerminalEmittedPayload {
  FinishReason reason = FinishReason::kLength;
};

struct RequestEvent {
  RequestEventKind kind;
  // Exactly one alternative is active; KindOf(event) == kind is checked in
  // debug builds and by tests.
  std::variant<StartTokenizationPayload, InputReadyPayload, TokenizationSucceededPayload,
               TokenizationFailedPayload, BeginReservationPayload, ReservationGrantedPayload,
               ReservationDeferredPayload, SubmitPayload, PrefillCompletedPayload,
               DecodeCompletedPayload, ExecutionFailedPayload, TerminalOutcomePayload,
               PreemptPayload, RequeuePayload, InFlightDrainedPayload, TerminalEmittedPayload>
      payload;
};

// Primary kind of the active payload alternative (unambiguous kinds only;
// submit disambiguates through its WorkKind, and shared outcome payloads
// report kCancelRequested as primary).
[[nodiscard]] RequestEventKind KindOf(const RequestEvent& event);

// True when the event's kind is compatible with its active payload
// alternative — the debug/test pairing check.
[[nodiscard]] bool KindMatchesPayload(RequestEventKind kind, const RequestEvent& event);

}  // namespace inferx

#endif  // INFERX_ENGINE_REQUEST_EVENT_H_
