#include "inferx/engine/request_controller.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/api/generate_request.h"
#include "inferx/api/response_event.h"
#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"
#include "inferx/engine/execution_completion.h"
#include "inferx/engine/request_state_machine.h"
#include "inferx/scheduler/work_kind.h"

namespace inferx {

// Prepared state: the decision plus every value Commit will move. Payload
// facts (step/ticket ids, token counts) are extracted here so Commit never
// touches the event or allocates.
struct PreparedTransition::Impl {
  TransitionDecision decision;
  TerminalResponse terminal_record;  // valid when decision.terminal
  RequestEventKind kind = RequestEventKind::kFatalError;
  StepId step{0};               // submit events
  ExecutionTicketId ticket{0};  // submit events
  ExecutionTicket submitted;    // submit events
  WorkKind work = WorkKind::kPrefill;
  TokenRange scheduled_range{TokenOffset(0), TokenOffset(0)};
  uint32_t scheduled_tokens = 0;  // submit/completion events
  uint32_t computed_delta = 0;    // successful completions
  uint32_t committed_delta = 0;   // synthetic outputs committed
  TokenId output_token{0};
  bool has_output_token = false;
  std::optional<ReservationId> reservation;
  TerminalResponse pending_terminal;
  bool has_pending_terminal = false;
  std::vector<TokenId> input_tokens;
};

PreparedTransition::PreparedTransition(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

PreparedTransition::PreparedTransition(PreparedTransition&&) noexcept = default;
PreparedTransition& PreparedTransition::operator=(PreparedTransition&&) noexcept = default;
PreparedTransition::~PreparedTransition() = default;

absl::StatusOr<PreparedTransition> RequestController::Prepare(const RequestContext& request,
                                                              const RequestEvent& event) const {
  try {
    absl::StatusOr<TransitionDecision> decision = DecideTransition(request, event);
    if (!decision.ok()) {
      return decision.status();
    }

    auto impl = std::make_unique<PreparedTransition::Impl>();
    impl->decision = *decision;
    impl->kind = event.kind;

    switch (event.kind) {
      case RequestEventKind::kSubmitPrefill:
      case RequestEventKind::kSubmitDecode: {
        const SubmitPayload& submit = std::get<SubmitPayload>(event.payload);
        impl->step = submit.step;
        impl->ticket = submit.ticket;
        impl->work = submit.work;
        impl->scheduled_range = submit.scheduled_range;
        impl->scheduled_tokens =
            submit.scheduled_range.size().ok() ? submit.scheduled_range.size()->value() : 0;
        impl->submitted = ExecutionTicket{submit.ticket, submit.step, submit.item_count};
        break;
      }
      case RequestEventKind::kPrefillCompleted:
      case RequestEventKind::kDecodeCompleted:
      case RequestEventKind::kInFlightDrained: {
        const ExecutionCompletion& completion =
            event.kind == RequestEventKind::kPrefillCompleted
                ? std::get<PrefillCompletedPayload>(event.payload).completion
            : event.kind == RequestEventKind::kDecodeCompleted
                ? std::get<DecodeCompletedPayload>(event.payload).completion
                : std::get<InFlightDrainedPayload>(event.payload).completion;
        impl->scheduled_tokens = completion.scheduled_tokens.size().value_or(TokenCount(0)).value();
        if (event.kind != RequestEventKind::kInFlightDrained && completion.status.ok()) {
          impl->computed_delta = impl->scheduled_tokens;
          impl->committed_delta = 1;
          impl->output_token = completion.output_token.value_or(TokenId(0));
          impl->has_output_token = completion.output_token.has_value();
        }
        break;
      }
      case RequestEventKind::kReservationGranted:
        impl->reservation = std::get<ReservationGrantedPayload>(event.payload).reservation;
        break;
      case RequestEventKind::kInputReady:
        impl->input_tokens = std::get<InputReadyPayload>(event.payload).tokens;
        break;
      case RequestEventKind::kTokenizationSucceeded:
        impl->input_tokens = std::get<TokenizationSucceededPayload>(event.payload).tokens;
        break;
      default:
        break;
    }

    if ((event.kind == RequestEventKind::kCancelRequested ||
         event.kind == RequestEventKind::kDeadlineExpired) &&
        !decision->terminal.has_value()) {
      const TerminalOutcomePayload& pending = std::get<TerminalOutcomePayload>(event.payload);
      impl->pending_terminal = TerminalResponse{
          .request = request.request->id,
          .reason = pending.finish,
          .status = pending.status.code(),
          .error_reason = pending.reason,
          .prompt_tokens = TokenCount(static_cast<uint32_t>(request.prompt_tokens.size())),
          .output_tokens = TokenCount(request.num_committed_output_tokens),
          .terminal_time = request.arrival.arrival_time,
      };
      impl->has_pending_terminal = true;
    }

    const std::optional<TerminalOutcome>& outcome = decision->terminal;
    if (outcome.has_value()) {
      // Materialize the terminal record here (the fallible step); Commit only
      // moves it. The coordinator stamps the terminal timestamp.
      impl->terminal_record = request.pending_terminal_reason.value_or(TerminalResponse{
          .request = request.request->id,
          .reason = outcome->reason,
          .status = outcome->status,
          .error_reason = outcome->error_reason,
          .prompt_tokens = TokenCount(static_cast<uint32_t>(request.prompt_tokens.size())),
          .output_tokens = TokenCount(request.num_committed_output_tokens),
          .terminal_time = request.arrival.arrival_time,
      });
    }
    return PreparedTransition(std::move(impl));
  } catch (const std::bad_alloc&) {
    return WithErrorReason(
        absl::ResourceExhaustedError("lifecycle.request: preparation allocation"),
        ErrorReason::kCapacityExhausted);
  } catch (...) {
    return WithErrorReason(absl::InternalError("lifecycle.request: preparation exception"),
                           ErrorReason::kInvariantViolation);
  }
}

void RequestController::StampTerminalTime(PreparedTransition& transition,
                                          MonotonicTime now) const noexcept {
  if (transition.impl_->decision.terminal.has_value()) {
    transition.impl_->terminal_record.terminal_time = now;
  }
  if (transition.impl_->has_pending_terminal) {
    transition.impl_->pending_terminal.terminal_time = now;
  }
}

void RequestController::Commit(RequestContext& request, PreparedTransition transition) noexcept {
  const TransitionDecision& decision = transition.impl_->decision;

  // Event-specific bookkeeping first (counters, in-flight ownership).
  switch (transition.impl_->kind) {
    case RequestEventKind::kInputReady:
    case RequestEventKind::kTokenizationSucceeded:
      request.prompt_tokens = std::move(transition.impl_->input_tokens);
      break;
    case RequestEventKind::kReservationGranted:
      request.reservation = transition.impl_->reservation;
      break;
    case RequestEventKind::kSubmitPrefill:
    case RequestEventKind::kSubmitDecode: {
      request.in_flight_step = transition.impl_->step;
      request.in_flight_ticket = transition.impl_->ticket;
      request.submitted_ticket = transition.impl_->submitted;
      request.in_flight_work = transition.impl_->work;
      request.in_flight_range = transition.impl_->scheduled_range;
      request.num_scheduled_tokens += transition.impl_->scheduled_tokens;
      break;
    }
    case RequestEventKind::kPrefillCompleted:
    case RequestEventKind::kDecodeCompleted: {
      request.num_scheduled_tokens -= transition.impl_->scheduled_tokens;
      request.num_computed_tokens += transition.impl_->computed_delta;
      request.num_committed_output_tokens += transition.impl_->committed_delta;
      if (transition.impl_->has_output_token) {
        request.output_tokens.push_back(transition.impl_->output_token);
      }
      request.in_flight_step.reset();
      request.in_flight_ticket.reset();
      request.submitted_ticket.reset();
      request.in_flight_work.reset();
      request.in_flight_range.reset();
      break;
    }
    // Drain (cancellation) and execution failure clear in-flight bookkeeping
    // without committing tokens; drain additionally proves no commit.
    case RequestEventKind::kInFlightDrained:
    case RequestEventKind::kExecutionFailed: {
      request.in_flight_step.reset();
      request.in_flight_ticket.reset();
      request.submitted_ticket.reset();
      request.in_flight_work.reset();
      request.in_flight_range.reset();
      request.num_scheduled_tokens = 0;
      break;
    }
    case RequestEventKind::kRequeue: {
      // Epoch overflow was rejected during Prepare's caller transaction;
      // Commit is noexcept.
      if (request.epoch.value() != std::numeric_limits<uint32_t>::max()) {
        request.epoch = RequestEpoch(request.epoch.value() + 1);
      }
      break;
    }
    default:
      break;
  }

  if (transition.impl_->has_pending_terminal) {
    request.pending_terminal_reason = transition.impl_->pending_terminal;
  }

  // State + terminal bookkeeping.
  request.state = decision.next;
  if (decision.terminal.has_value()) {
    request.terminal = transition.impl_->terminal_record;
    // Exactly-once marker: entering a terminal state and building the
    // terminal response is one lifecycle commit.
    request.terminal_emitted = true;
    request.pending_terminal_reason.reset();
    request.num_scheduled_tokens = 0;
  }
  if (decision.effects & TransitionEffect::kReleaseReservation) {
    request.reservation.reset();
  }
}

}  // namespace inferx
