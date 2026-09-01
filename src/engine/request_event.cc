#include "inferx/engine/request_event.h"

#include <type_traits>

#include "inferx/engine/execution_ticket.h"

namespace inferx {

absl::string_view ToString(RequestEventKind kind) {
  switch (kind) {
    case RequestEventKind::kStartTokenization:
      return "start_tokenization";
    case RequestEventKind::kInputReady:
      return "input_ready";
    case RequestEventKind::kTokenizationSucceeded:
      return "tokenization_succeeded";
    case RequestEventKind::kTokenizationFailed:
      return "tokenization_failed";
    case RequestEventKind::kBeginReservation:
      return "begin_reservation";
    case RequestEventKind::kReservationGranted:
      return "reservation_granted";
    case RequestEventKind::kReservationDeferred:
      return "reservation_deferred";
    case RequestEventKind::kSubmitPrefill:
      return "submit_prefill";
    case RequestEventKind::kPrefillCompleted:
      return "prefill_completed";
    case RequestEventKind::kSubmitDecode:
      return "submit_decode";
    case RequestEventKind::kDecodeCompleted:
      return "decode_completed";
    case RequestEventKind::kPreempt:
      return "preempt";
    case RequestEventKind::kRequeue:
      return "requeue";
    case RequestEventKind::kCancelRequested:
      return "cancel_requested";
    case RequestEventKind::kDeadlineExpired:
      return "deadline_expired";
    case RequestEventKind::kInFlightDrained:
      return "in_flight_drained";
    case RequestEventKind::kExecutionFailed:
      return "execution_failed";
    case RequestEventKind::kFatalError:
      return "fatal_error";
    case RequestEventKind::kTerminalEmitted:
      return "terminal_emitted";
  }
  return "unknown";
}

std::optional<RequestEventKind> RequestEventKindFromName(absl::string_view name) {
  for (uint32_t value = 0; value <= static_cast<uint32_t>(RequestEventKind::kTerminalEmitted);
       ++value) {
    const auto kind = static_cast<RequestEventKind>(value);
    if (ToString(kind) == name) {
      return kind;
    }
  }
  return std::nullopt;
}

RequestEventKind KindOf(const RequestEvent& event) {
  return std::visit(
      [](const auto& payload) -> RequestEventKind {
        using Payload = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<Payload, StartTokenizationPayload>) {
          return RequestEventKind::kStartTokenization;
        } else if constexpr (std::is_same_v<Payload, InputReadyPayload>) {
          return RequestEventKind::kInputReady;
        } else if constexpr (std::is_same_v<Payload, TokenizationSucceededPayload>) {
          return RequestEventKind::kTokenizationSucceeded;
        } else if constexpr (std::is_same_v<Payload, TokenizationFailedPayload>) {
          return RequestEventKind::kTokenizationFailed;
        } else if constexpr (std::is_same_v<Payload, BeginReservationPayload>) {
          return RequestEventKind::kBeginReservation;
        } else if constexpr (std::is_same_v<Payload, ReservationGrantedPayload>) {
          return RequestEventKind::kReservationGranted;
        } else if constexpr (std::is_same_v<Payload, ReservationDeferredPayload>) {
          return RequestEventKind::kReservationDeferred;
        } else if constexpr (std::is_same_v<Payload, SubmitPayload>) {
          return payload.work == WorkKind::kPrefill ? RequestEventKind::kSubmitPrefill
                                                    : RequestEventKind::kSubmitDecode;
        } else if constexpr (std::is_same_v<Payload, PrefillCompletedPayload>) {
          return RequestEventKind::kPrefillCompleted;
        } else if constexpr (std::is_same_v<Payload, DecodeCompletedPayload>) {
          return RequestEventKind::kDecodeCompleted;
        } else if constexpr (std::is_same_v<Payload, ExecutionFailedPayload>) {
          return RequestEventKind::kExecutionFailed;
        } else if constexpr (std::is_same_v<Payload, TerminalOutcomePayload>) {
          return RequestEventKind::kCancelRequested;
        } else if constexpr (std::is_same_v<Payload, PreemptPayload>) {
          return RequestEventKind::kPreempt;
        } else if constexpr (std::is_same_v<Payload, RequeuePayload>) {
          return RequestEventKind::kRequeue;
        } else if constexpr (std::is_same_v<Payload, InFlightDrainedPayload>) {
          return RequestEventKind::kInFlightDrained;
        } else if constexpr (std::is_same_v<Payload, TerminalEmittedPayload>) {
          return RequestEventKind::kTerminalEmitted;
        }
      },
      event.payload);
}

bool KindMatchesPayload(RequestEventKind kind, const RequestEvent& event) {
  const RequestEventKind primary = KindOf(event);
  if (primary == kind) {
    return true;
  }
  // Shared alternatives: cancel/deadline/fatal share the outcome payload;
  // both submit kinds share SubmitPayload (distinguished by WorkKind, which
  // KindOf already resolved, so a mismatch here is a real mismatch).
  return primary == RequestEventKind::kCancelRequested &&
         (kind == RequestEventKind::kDeadlineExpired || kind == RequestEventKind::kFatalError);
}

}  // namespace inferx
