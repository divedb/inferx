#include "inferx/engine/request_state_machine.h"

#include <array>
#include <cstddef>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/status.h"
#include "inferx/engine/execution_ticket.h"

namespace inferx {

namespace {

// --- Named pure resolvers for conditional destinations -----------------------

// ReservationGranted: nothing computed yet -> prefill; otherwise decode.
RequestState ResolveAfterReservation(const RequestContext& context, const RequestEvent& event) {
  (void)event;
  return context.num_computed_tokens == 0 ? RequestState::kPrefillReady
                                          : RequestState::kDecodeReady;
}

// Prefill/decode completion: finishing when no output remains, counting the
// outputs this completion itself commits (context is pre-commit).
RequestState ResolveAfterCompletion(const RequestContext& context, const RequestEvent& event) {
  uint32_t committed = context.num_committed_output_tokens;
  if (const auto* prefill = std::get_if<PrefillCompletedPayload>(&event.payload)) {
    committed += prefill->completion.status.ok() ? 1U : 0U;
  } else if (const auto* decode = std::get_if<DecodeCompletedPayload>(&event.payload)) {
    committed += decode->completion.status.ok() ? 1U : 0U;
  }
  return committed < context.request->generation.max_output_tokens.value()
             ? RequestState::kDecodeReady
             : RequestState::kFinishing;
}

// --- The expanded declarative table -----------------------------------------
// Wildcard rows are expanded; count kept explicit so a
// drifted table fails to compile.
inline constexpr size_t kRuleCount = 45;  // Finishing row was already counted

constexpr std::array<TransitionRule, kRuleCount> kRules = {{
    {RequestState::kReceived, RequestEventKind::kStartTokenization, nullptr,
     RequestState::kTokenizing},
    {RequestState::kReceived, RequestEventKind::kInputReady, nullptr, RequestState::kQueued},
    {RequestState::kTokenizing, RequestEventKind::kTokenizationSucceeded, nullptr,
     RequestState::kQueued},
    {RequestState::kTokenizing, RequestEventKind::kTokenizationFailed, nullptr,
     RequestState::kFailed, TransitionEffect::kNone, true},
    {RequestState::kQueued, RequestEventKind::kBeginReservation, nullptr, RequestState::kReserving},
    {RequestState::kReserving, RequestEventKind::kReservationGranted, ResolveAfterReservation,
     RequestState::kReceived},
    {RequestState::kReserving, RequestEventKind::kReservationDeferred, nullptr,
     RequestState::kQueued},
    {RequestState::kPrefillReady, RequestEventKind::kSubmitPrefill, nullptr,
     RequestState::kPrefilling},
    {RequestState::kPrefilling, RequestEventKind::kPrefillCompleted, ResolveAfterCompletion,
     RequestState::kReceived, TransitionEffect::kClearInFlight},
    {RequestState::kDecodeReady, RequestEventKind::kSubmitDecode, nullptr, RequestState::kDecoding},
    // A committed output matched a stop id: finish successfully from a ready
    // state (M5 schema extension, m5.md section 13.4). The reservation
    // releases at terminal emission like every other finish.
    {RequestState::kPrefillReady, RequestEventKind::kStopMatched, nullptr,
     RequestState::kFinishing},
    {RequestState::kDecodeReady, RequestEventKind::kStopMatched, nullptr, RequestState::kFinishing},
    {RequestState::kDecoding, RequestEventKind::kDecodeCompleted, ResolveAfterCompletion,
     RequestState::kReceived, TransitionEffect::kClearInFlight},
    {RequestState::kPrefillReady, RequestEventKind::kPreempt, nullptr, RequestState::kPreempted,
     TransitionEffect::kReleaseReservation},
    {RequestState::kDecodeReady, RequestEventKind::kPreempt, nullptr, RequestState::kPreempted,
     TransitionEffect::kReleaseReservation},
    {RequestState::kPreempted, RequestEventKind::kRequeue, nullptr, RequestState::kQueued,
     TransitionEffect::kIncrementEpoch},
    // Cancel/deadline from every nonterminal non-in-flight state except
    // Finishing (wildcard expanded).
    {RequestState::kReceived, RequestEventKind::kCancelRequested, nullptr, RequestState::kCancelled,
     TransitionEffect::kNone, true},
    {RequestState::kReceived, RequestEventKind::kDeadlineExpired, nullptr, RequestState::kCancelled,
     TransitionEffect::kNone, true},
    {RequestState::kTokenizing, RequestEventKind::kCancelRequested, nullptr,
     RequestState::kCancelled, TransitionEffect::kNone, true},
    {RequestState::kTokenizing, RequestEventKind::kDeadlineExpired, nullptr,
     RequestState::kCancelled, TransitionEffect::kNone, true},
    {RequestState::kQueued, RequestEventKind::kCancelRequested, nullptr, RequestState::kCancelled,
     TransitionEffect::kNone, true},
    {RequestState::kQueued, RequestEventKind::kDeadlineExpired, nullptr, RequestState::kCancelled,
     TransitionEffect::kNone, true},
    {RequestState::kReserving, RequestEventKind::kCancelRequested, nullptr,
     RequestState::kCancelled, TransitionEffect::kNone, true},
    {RequestState::kReserving, RequestEventKind::kDeadlineExpired, nullptr,
     RequestState::kCancelled, TransitionEffect::kNone, true},
    {RequestState::kPrefillReady, RequestEventKind::kCancelRequested, nullptr,
     RequestState::kCancelled, TransitionEffect::kReleaseReservation, true},
    {RequestState::kPrefillReady, RequestEventKind::kDeadlineExpired, nullptr,
     RequestState::kCancelled, TransitionEffect::kReleaseReservation, true},
    {RequestState::kDecodeReady, RequestEventKind::kCancelRequested, nullptr,
     RequestState::kCancelled, TransitionEffect::kReleaseReservation, true},
    {RequestState::kDecodeReady, RequestEventKind::kDeadlineExpired, nullptr,
     RequestState::kCancelled, TransitionEffect::kReleaseReservation, true},
    {RequestState::kPreempted, RequestEventKind::kCancelRequested, nullptr,
     RequestState::kCancelled, TransitionEffect::kNone, true},
    {RequestState::kPreempted, RequestEventKind::kDeadlineExpired, nullptr,
     RequestState::kCancelled, TransitionEffect::kNone, true},
    // In-flight cancellation: drain first, release on terminal commit.
    {RequestState::kPrefilling, RequestEventKind::kCancelRequested, nullptr,
     RequestState::kCancelling},
    {RequestState::kPrefilling, RequestEventKind::kDeadlineExpired, nullptr,
     RequestState::kCancelling},
    {RequestState::kDecoding, RequestEventKind::kCancelRequested, nullptr,
     RequestState::kCancelling},
    {RequestState::kDecoding, RequestEventKind::kDeadlineExpired, nullptr,
     RequestState::kCancelling},
    {RequestState::kCancelling, RequestEventKind::kInFlightDrained, nullptr,
     RequestState::kCancelled,
     TransitionEffect::kClearInFlight | TransitionEffect::kReleaseReservation, true},
    // Execution failure completes in-flight states terminally.
    {RequestState::kPrefilling, RequestEventKind::kExecutionFailed, nullptr, RequestState::kFailed,
     TransitionEffect::kClearInFlight | TransitionEffect::kReleaseReservation, true},
    {RequestState::kDecoding, RequestEventKind::kExecutionFailed, nullptr, RequestState::kFailed,
     TransitionEffect::kClearInFlight | TransitionEffect::kReleaseReservation, true},
    // Fatal events from any nonterminal non-in-flight state.
    {RequestState::kReceived, RequestEventKind::kFatalError, nullptr, RequestState::kFailed,
     TransitionEffect::kNone, true},
    {RequestState::kTokenizing, RequestEventKind::kFatalError, nullptr, RequestState::kFailed,
     TransitionEffect::kNone, true},
    {RequestState::kQueued, RequestEventKind::kFatalError, nullptr, RequestState::kFailed,
     TransitionEffect::kNone, true},
    {RequestState::kReserving, RequestEventKind::kFatalError, nullptr, RequestState::kFailed,
     TransitionEffect::kNone, true},
    {RequestState::kPrefillReady, RequestEventKind::kFatalError, nullptr, RequestState::kFailed,
     TransitionEffect::kReleaseReservation, true},
    {RequestState::kDecodeReady, RequestEventKind::kFatalError, nullptr, RequestState::kFailed,
     TransitionEffect::kReleaseReservation, true},
    {RequestState::kPreempted, RequestEventKind::kFatalError, nullptr, RequestState::kFailed,
     TransitionEffect::kNone, true},
    {RequestState::kFinishing, RequestEventKind::kTerminalEmitted, nullptr, RequestState::kFinished,
     TransitionEffect::kReleaseReservation, true},
}};

static_assert(kRules.size() == kRuleCount, "rule count drift");

absl::Status InvalidTransition(const RequestContext& request, RequestEventKind kind) {
  return WithErrorReason(
      absl::Status(absl::StatusCode::kFailedPrecondition,
                   absl::StrCat("lifecycle.request.", ToString(request.state), ": event ",
                                ToString(kind), " is not legal in this state")),
      ErrorReason::kInvalidTransition);
}

bool MatchesSubmittedWork(const RequestContext& context, const ExecutionCompletion& completion) {
  return context.submitted_ticket.has_value() && context.in_flight_work.has_value() &&
         context.in_flight_range.has_value() &&
         TicketMatches(*context.submitted_ticket, completion) &&
         context.request->id == completion.request && context.sequence == completion.sequence &&
         context.epoch == completion.epoch && *context.in_flight_work == completion.kind &&
         context.in_flight_range->begin == completion.scheduled_tokens.begin &&
         context.in_flight_range->end == completion.scheduled_tokens.end &&
         !context.terminal_emitted;
}

}  // namespace

TransitionRuleSpan TransitionRules() { return {kRules.data(), kRules.size()}; }

absl::StatusOr<TransitionDecision> DecideTransition(const RequestContext& request,
                                                    const RequestEvent& event) {
  if (IsTerminal(request.state)) {
    return InvalidTransition(request, event.kind);
  }

  const TransitionRule* matched = nullptr;
  for (const TransitionRule& rule : kRules) {
    if (rule.current == request.state && rule.event == event.kind) {
      matched = &rule;
      break;
    }
  }
  if (matched == nullptr) {
    return InvalidTransition(request, event.kind);
  }

  // --- Explicit guards on data-bearing events ------------------------------
  switch (event.kind) {
    case RequestEventKind::kInputReady:
      if (std::get_if<InputReadyPayload>(&event.payload) == nullptr ||
          std::get<InputReadyPayload>(event.payload).tokens.empty()) {
        return InvalidTransition(request, event.kind);
      }
      break;
    case RequestEventKind::kTokenizationSucceeded:
      if (std::get_if<TokenizationSucceededPayload>(&event.payload) == nullptr ||
          std::get<TokenizationSucceededPayload>(event.payload).tokens.empty()) {
        return InvalidTransition(request, event.kind);
      }
      break;
    case RequestEventKind::kSubmitPrefill:
    case RequestEventKind::kSubmitDecode: {
      const SubmitPayload* submit = std::get_if<SubmitPayload>(&event.payload);
      const bool expected_kind = event.kind == RequestEventKind::kSubmitPrefill
                                     ? submit != nullptr && submit->work == WorkKind::kPrefill
                                     : submit != nullptr && submit->work == WorkKind::kDecode;
      if (submit == nullptr || !expected_kind || submit->epoch != request.epoch ||
          request.in_flight_step.has_value() || !request.reservation.has_value() ||
          submit->item_count == 0 || !submit->scheduled_range.Validate().ok() ||
          submit->scheduled_range.begin == submit->scheduled_range.end) {
        return InvalidTransition(request, event.kind);
      }
      break;
    }
    case RequestEventKind::kReservationGranted: {
      const auto* granted = std::get_if<ReservationGrantedPayload>(&event.payload);
      if (granted == nullptr || request.reservation.has_value() ||
          granted->cost.sequences.value() == 0 || granted->cost.kv_tokens.value() == 0) {
        return InvalidTransition(request, event.kind);
      }
      break;
    }
    case RequestEventKind::kPrefillCompleted:
    case RequestEventKind::kDecodeCompleted:
    case RequestEventKind::kInFlightDrained:
    case RequestEventKind::kExecutionFailed: {
      const ExecutionCompletion* completion = nullptr;
      if (const auto* prefill = std::get_if<PrefillCompletedPayload>(&event.payload)) {
        completion = &prefill->completion;
      } else if (const auto* decode = std::get_if<DecodeCompletedPayload>(&event.payload)) {
        completion = &decode->completion;
      } else if (const auto* failed = std::get_if<ExecutionFailedPayload>(&event.payload)) {
        completion = &failed->completion;
      } else if (const auto* drained = std::get_if<InFlightDrainedPayload>(&event.payload)) {
        completion = &drained->completion;
      }
      if (completion == nullptr || !MatchesSubmittedWork(request, *completion)) {
        return WithErrorReason(
            absl::Status(absl::StatusCode::kFailedPrecondition,
                         absl::StrCat("lifecycle.request.", ToString(request.state),
                                      ": stale or mismatched completion")),
            ErrorReason::kStaleCompletion);
      }
      if ((event.kind == RequestEventKind::kPrefillCompleted ||
           event.kind == RequestEventKind::kDecodeCompleted) &&
          (!completion->status.ok() || !completion->output_token.has_value())) {
        return InvalidTransition(request, event.kind);
      }
      if (event.kind == RequestEventKind::kExecutionFailed && completion->status.ok()) {
        return InvalidTransition(request, event.kind);
      }
      break;
    }
    default:
      break;
  }

  TransitionDecision decision;
  decision.next = matched->resolve_next != nullptr ? matched->resolve_next(request, event)
                                                   : matched->fixed_next;
  decision.effects = matched->effects;

  if (matched->terminal) {
    TerminalOutcome outcome;
    if (const auto* terminal = std::get_if<TerminalOutcomePayload>(&event.payload)) {
      outcome.reason = terminal->finish;
      outcome.status = terminal->status.code();
      outcome.error_reason = terminal->reason;
    } else if (event.kind == RequestEventKind::kTokenizationFailed) {
      const auto& failure = std::get<TokenizationFailedPayload>(event.payload);
      outcome.reason = FinishReason::kExecutorError;
      outcome.status = failure.failure.code();
      outcome.error_reason = GetErrorReason(failure.failure).value_or(ErrorReason::kInvalidRequest);
    } else if (event.kind == RequestEventKind::kExecutionFailed) {
      const auto& failed = std::get<ExecutionFailedPayload>(event.payload);
      outcome.reason = FinishReason::kExecutorError;
      outcome.status = failed.completion.status.code();
      outcome.error_reason = failed.completion.error_reason;
    } else if (event.kind == RequestEventKind::kInFlightDrained &&
               request.pending_terminal_reason.has_value()) {
      outcome.reason = request.pending_terminal_reason->reason;
      outcome.status = request.pending_terminal_reason->status;
      outcome.error_reason = request.pending_terminal_reason->error_reason;
    } else if (event.kind == RequestEventKind::kDeadlineExpired) {
      outcome.reason = FinishReason::kDeadline;
      outcome.status = absl::StatusCode::kDeadlineExceeded;
      outcome.error_reason = ErrorReason::kDeadlineExpired;
    } else if (event.kind == RequestEventKind::kTerminalEmitted) {
      outcome.reason = std::get<TerminalEmittedPayload>(event.payload).reason;
      outcome.status = absl::StatusCode::kOk;
      outcome.error_reason = std::nullopt;
    } else if (event.kind == RequestEventKind::kCancelRequested) {
      outcome.reason = FinishReason::kCancelled;
      outcome.status = absl::StatusCode::kCancelled;
      outcome.error_reason = ErrorReason::kExplicitCancellation;
    } else {
      outcome.reason = FinishReason::kExecutorError;
      outcome.status = absl::StatusCode::kInternal;
      outcome.error_reason = ErrorReason::kInvariantViolation;
    }
    decision.terminal = outcome;
  }
  return decision;
}

}  // namespace inferx
