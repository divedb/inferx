#include "inferx/engine/request_controller.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "inferx/base/status.h"
#include "inferx/engine/execution_ticket.h"

namespace inferx {

// Prepared state: the decision plus every value Commit will move. Payload
// facts (step/ticket ids, token counts) are extracted here so Commit never
// touches the event or allocates.
struct PreparedTransition::Impl {
  TransitionDecision decision;
  TerminalResponse terminal_record;  // valid when decision.terminal
  RequestEventKind kind = RequestEventKind::kFatalError;
  StepId step{0};                 // submit events
  ExecutionTicketId ticket{0};    // submit events
  ExecutionTicket submitted;      // submit events
  uint32_t scheduled_tokens = 0;  // submit/completion events
  uint32_t computed_delta = 0;    // successful completions
  uint32_t committed_delta = 0;   // synthetic outputs committed
};

PreparedTransition::PreparedTransition(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

PreparedTransition::PreparedTransition(PreparedTransition&&) noexcept = default;
PreparedTransition& PreparedTransition::operator=(PreparedTransition&&) noexcept = default;
PreparedTransition::~PreparedTransition() = default;

absl::StatusOr<PreparedTransition> RequestController::Prepare(const RequestContext& request,
                                                              const RequestEvent& event) const {
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
      impl->scheduled_tokens =
          submit.scheduled_range.size().ok() ? submit.scheduled_range.size()->value() : 0;
      impl->submitted = ExecutionTicket{
          request.request->id,
          request.sequence,
          request.epoch,
          submit.step,
          submit.ticket,
          TokenCount(submit.scheduled_range.size().value_or(TokenCount(0)).value())};
      break;
    }
    case RequestEventKind::kPrefillCompleted:
    case RequestEventKind::kDecodeCompleted: {
      const ExecutionCompletion& completion =
          event.kind == RequestEventKind::kPrefillCompleted
              ? std::get<PrefillCompletedPayload>(event.payload).completion
              : std::get<DecodeCompletedPayload>(event.payload).completion;
      impl->scheduled_tokens = completion.ticket.scheduled_tokens.value();
      impl->computed_delta = completion.computed_tokens.value();
      impl->committed_delta = completion.committed_tokens.value();
      break;
    }
    default:
      break;
  }

  const std::optional<TerminalOutcome>& outcome = decision->terminal;
  if (outcome.has_value()) {
    // Materialize the terminal record here (the fallible step); Commit only
    // moves it. The coordinator stamps the terminal timestamp.
    impl->terminal_record = TerminalResponse{
        .request = request.request->id,
        .reason = outcome->reason,
        .status = outcome->status,
        .error_reason = outcome->error_reason,
        .prompt_tokens = TokenCount(static_cast<uint32_t>(request.prompt_tokens.size())),
        .output_tokens = TokenCount(request.num_committed_output_tokens),
        .terminal_time = request.arrival.arrival_time,
    };
  }
  return PreparedTransition(std::move(impl));
}

void RequestController::Commit(RequestContext& request, PreparedTransition transition) noexcept {
  const TransitionDecision& decision = transition.impl_->decision;

  // Event-specific bookkeeping first (counters, in-flight ownership).
  switch (transition.impl_->kind) {
    case RequestEventKind::kSubmitPrefill:
    case RequestEventKind::kSubmitDecode: {
      request.in_flight_step = transition.impl_->step;
      request.in_flight_ticket = transition.impl_->ticket;
      request.submitted_ticket = transition.impl_->submitted;
      request.num_scheduled_tokens += transition.impl_->scheduled_tokens;
      break;
    }
    case RequestEventKind::kPrefillCompleted:
    case RequestEventKind::kDecodeCompleted: {
      request.num_scheduled_tokens -= transition.impl_->scheduled_tokens;
      request.num_computed_tokens += transition.impl_->computed_delta;
      request.num_committed_output_tokens += transition.impl_->committed_delta;
      request.in_flight_step.reset();
      request.in_flight_ticket.reset();
      request.submitted_ticket.reset();
      break;
    }
    // Drain (cancellation) and execution failure clear in-flight bookkeeping
    // without committing tokens; drain additionally proves no commit.
    case RequestEventKind::kInFlightDrained:
    case RequestEventKind::kExecutionFailed: {
      request.in_flight_step.reset();
      request.in_flight_ticket.reset();
      request.submitted_ticket.reset();
      request.num_scheduled_tokens = 0;
      break;
    }
    case RequestEventKind::kRequeue: {
      // Epoch overflow was rejected during Prepare's caller transaction;
      // Commit is noexcept.
      if (request.epoch.value() != UINT32_MAX) {
        request.epoch = RequestEpoch(request.epoch.value() + 1);
      }
      break;
    }
    default:
      break;
  }

  // State + terminal bookkeeping.
  request.state = decision.next;
  if (decision.terminal.has_value()) {
    request.terminal = transition.impl_->terminal_record;
    // Exactly-once marker: entering a terminal state and building the
    // terminal response is one lifecycle commit (m1.md section 11.2).
    request.terminal_emitted = true;
    request.num_scheduled_tokens = 0;
  }
  if (decision.effects & TransitionEffect::kReleaseReservation) {
    request.reservation.reset();
  }
}

}  // namespace inferx
