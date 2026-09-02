#include "inferx/engine/lifecycle_coordinator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/status.h"
#include "inferx/engine/request_event.h"

namespace inferx {
namespace {

absl::Status UnknownRequest(RequestId request) {
  return WithErrorReason(
      absl::NotFoundError(absl::StrCat("lifecycle.request.", request.value(), ": unknown")),
      ErrorReason::kUnknownResource);
}

TerminalResponse TerminalFrom(const RequestContext& request, FinishReason finish,
                              absl::StatusCode status, std::optional<ErrorReason> reason,
                              MonotonicTime now) {
  return TerminalResponse{
      .request = request.request->id,
      .reason = finish,
      .status = status,
      .error_reason = reason,
      .prompt_tokens = TokenCount(static_cast<uint32_t>(request.prompt_tokens.size())),
      .output_tokens = TokenCount(request.num_committed_output_tokens),
      .terminal_time = now,
  };
}

}  // namespace

LifecycleCoordinator::LifecycleCoordinator(RequestRegistry& registry,
                                           scheduler::ResourceAccountant& resources,
                                           ResponseSink& responses, LifecycleObserver& observer,
                                           size_t max_live_requests) noexcept
    : registry_(registry),
      resources_(resources),
      responses_(responses),
      observer_(observer),
      max_live_requests_(max_live_requests) {}

absl::Status LifecycleCoordinator::Admit(GenerateRequest request, MonotonicTime arrival) {
  if (registry_.size() >= max_live_requests_) {
    return WithErrorReason(absl::ResourceExhaustedError("lifecycle.admit: registry is full"),
                           ErrorReason::kQueueFull);
  }
  if (registry_.Contains(request.id)) {
    return WithErrorReason(absl::AlreadyExistsError("lifecycle.admit: duplicate request id"),
                           ErrorReason::kDuplicateRequest);
  }
  const auto* tokens = std::get_if<std::vector<TokenId>>(&request.input);
  if (tokens == nullptr || tokens->empty()) {
    return WithErrorReason(absl::InvalidArgumentError("lifecycle.admit: token input required"),
                           ErrorReason::kInvalidRequest);
  }

  try {
    auto context = std::make_unique<RequestContext>();
    context->prompt_tokens.reserve(tokens->size());
    context->output_tokens.reserve(request.generation.max_output_tokens.value());
    context->arrival.arrival_time = arrival;
    context->request = std::make_unique<const GenerateRequest>(std::move(request));
    context->reservation_output_tokens = context->request->generation.max_output_tokens;
    const std::vector<TokenId>& owned_tokens =
        std::get<std::vector<TokenId>>(context->request->input);
    RequestEvent event{RequestEventKind::kInputReady, InputReadyPayload{owned_tokens}};
    absl::StatusOr<PreparedTransition> prepared = controller_.Prepare(*context, event);
    if (!prepared.ok()) {
      return prepared.status();
    }
    const RequestState from = context->state;
    controller_.Commit(*context, std::move(*prepared));
    const RequestId id = context->request->id;
    absl::Status inserted = registry_.Insert(std::move(context));
    if (!inserted.ok()) {
      return inserted;
    }
    return ObserveTransition(*registry_.Find(id), from, event.kind, arrival);
  } catch (const std::bad_alloc&) {
    return WithErrorReason(absl::ResourceExhaustedError("lifecycle.admit: allocation failed"),
                           ErrorReason::kQueueFull);
  } catch (...) {
    return WithErrorReason(absl::InternalError("lifecycle.admit: unexpected exception"),
                           ErrorReason::kInvariantViolation);
  }
}

absl::Status LifecycleCoordinator::Reserve(RequestId request_id, scheduler::ResourceCost cost,
                                           MonotonicTime now) {
  RequestContext* request = registry_.Find(request_id);
  if (request == nullptr) {
    return UnknownRequest(request_id);
  }
  const RequestState queued = request->state;
  RequestEvent begin{RequestEventKind::kBeginReservation, BeginReservationPayload{}};
  absl::StatusOr<PreparedTransition> begin_prepared = controller_.Prepare(*request, begin);
  if (!begin_prepared.ok()) {
    return begin_prepared.status();
  }
  controller_.Commit(*request, std::move(*begin_prepared));
  absl::Status observed = ObserveTransition(*request, queued, begin.kind, now);
  if (!observed.ok()) {
    return observed;
  }

  absl::StatusOr<scheduler::ReservationTxn> transaction =
      resources_.BeginReservation(request_id, cost);
  if (!transaction.ok()) {
    RequestEvent deferred{RequestEventKind::kReservationDeferred, ReservationDeferredPayload{}};
    absl::StatusOr<PreparedTransition> deferred_prepared = controller_.Prepare(*request, deferred);
    if (!deferred_prepared.ok()) {
      return deferred_prepared.status();
    }
    const RequestState reserving = request->state;
    controller_.Commit(*request, std::move(*deferred_prepared));
    absl::Status deferred_observed = ObserveTransition(*request, reserving, deferred.kind, now);
    return deferred_observed.ok() ? transaction.status() : deferred_observed;
  }

  RequestEvent granted{
      RequestEventKind::kReservationGranted,
      ReservationGrantedPayload{transaction->id(), transaction->cost()},
  };
  absl::StatusOr<PreparedTransition> granted_prepared = controller_.Prepare(*request, granted);
  if (!granted_prepared.ok()) {
    return granted_prepared.status();
  }
  const RequestState reserving = request->state;
  const ReservationId reservation = transaction->Commit();
  controller_.Commit(*request, std::move(*granted_prepared));
  absl::Status resource_observed = observer_.ObserveResource(ResourceObservation{
      .at = now,
      .action = ResourceAction::kReserve,
      .reservation = reservation,
      .request = request_id,
      .cost = cost,
      .snapshot = resources_.Snapshot(),
  });
  if (!resource_observed.ok()) {
    return resource_observed;
  }
  return ObserveTransition(*request, reserving, granted.kind, now);
}

absl::Status LifecycleCoordinator::MarkSubmitted(const scheduler::StepPlan& plan,
                                                 const ExecutionTicket& ticket, MonotonicTime now) {
  if (ticket.step != plan.id || ticket.item_count != plan.sequences.size()) {
    return WithErrorReason(
        absl::InvalidArgumentError("lifecycle.submit: ticket does not match accepted plan"),
        ErrorReason::kExecutorRejected);
  }
  std::vector<PreparedTransition> prepared;
  std::vector<RequestContext*> requests;
  std::vector<RequestState> from_states;
  try {
    prepared.reserve(plan.sequences.size());
    requests.reserve(plan.sequences.size());
    from_states.reserve(plan.sequences.size());
  } catch (...) {
    return WithErrorReason(
        absl::ResourceExhaustedError("lifecycle.submit: transition scratch allocation failed"),
        ErrorReason::kCapacityExhausted);
  }
  for (const scheduler::ScheduledSequence& item : plan.sequences) {
    RequestContext* request = registry_.Find(item.request);
    if (request == nullptr) {
      return UnknownRequest(item.request);
    }
    const RequestEventKind kind = item.kind == WorkKind::kPrefill ? RequestEventKind::kSubmitPrefill
                                                                  : RequestEventKind::kSubmitDecode;
    RequestEvent event{kind, SubmitPayload{.step = ticket.step,
                                           .ticket = ticket.id,
                                           .epoch = item.epoch,
                                           .work = item.kind,
                                           .scheduled_range = item.input_tokens,
                                           .item_count = ticket.item_count}};
    absl::StatusOr<PreparedTransition> transition = controller_.Prepare(*request, event);
    if (!transition.ok()) {
      return transition.status();
    }
    requests.push_back(request);
    from_states.push_back(request->state);
    prepared.push_back(std::move(*transition));
  }
  for (size_t index = 0; index < prepared.size(); ++index) {
    const RequestEventKind kind = plan.sequences[index].kind == WorkKind::kPrefill
                                      ? RequestEventKind::kSubmitPrefill
                                      : RequestEventKind::kSubmitDecode;
    controller_.Commit(*requests[index], std::move(prepared[index]));
    absl::Status observed = ObserveTransition(*requests[index], from_states[index], kind, now);
    if (!observed.ok()) {
      return observed;
    }
  }
  return absl::OkStatus();
}

absl::Status LifecycleCoordinator::Complete(const ExecutionCompletion& completion,
                                            MonotonicTime now) {
  RequestContext* request = registry_.Find(completion.request);
  if (request == nullptr) {
    return UnknownRequest(completion.request);
  }
  RequestEvent event =
      request->state == RequestState::kCancelling
          ? RequestEvent{RequestEventKind::kInFlightDrained, InFlightDrainedPayload{completion}}
      : !completion.status.ok()
          ? RequestEvent{RequestEventKind::kExecutionFailed, ExecutionFailedPayload{completion}}
      : completion.kind == WorkKind::kPrefill
          ? RequestEvent{RequestEventKind::kPrefillCompleted, PrefillCompletedPayload{completion}}
          : RequestEvent{RequestEventKind::kDecodeCompleted, DecodeCompletedPayload{completion}};
  absl::StatusOr<PreparedTransition> prepared = controller_.Prepare(*request, event);
  if (!prepared.ok()) {
    return prepared.status();
  }
  controller_.StampTerminalTime(*prepared, now);

  const bool terminal = request->state == RequestState::kCancelling || !completion.status.ok();
  std::optional<ResponseReservation> response;
  if (terminal) {
    const TerminalResponse terminal_response =
        request->state == RequestState::kCancelling && request->pending_terminal_reason.has_value()
            ? *request->pending_terminal_reason
            : TerminalFrom(*request, FinishReason::kExecutorError, completion.status.code(),
                           completion.error_reason, now);
    auto reserved = responses_.Prepare(ResponseEvent{terminal_response});
    if (!reserved.ok()) {
      return reserved.status();
    }
    response = std::move(*reserved);
  } else {
    auto reserved = responses_.Prepare(ResponseEvent{TokenDelta{
        .request = completion.request,
        .token = *completion.output_token,
        .output_position = TokenOffset(request->num_committed_output_tokens),
    }});
    if (!reserved.ok()) {
      return reserved.status();
    }
    response = std::move(*reserved);
  }

  const RequestState from = request->state;
  std::optional<ResourceObservation> release_observation;
  if (terminal && request->reservation.has_value()) {
    absl::StatusOr<ResourceObservation> released = ReleaseReservation(*request, now);
    if (!released.ok()) {
      return released.status();
    }
    release_observation = *released;
  }
  controller_.Commit(*request, std::move(*prepared));
  responses_.Commit(std::move(*response));
  if (release_observation.has_value()) {
    absl::Status observed_resource = observer_.ObserveResource(*release_observation);
    if (!observed_resource.ok()) {
      return observed_resource;
    }
  }
  return ObserveTransition(*request, from, event.kind, now);
}

absl::Status LifecycleCoordinator::Stop(RequestId request_id, MonotonicTime now) {
  RequestContext* request = registry_.Find(request_id);
  if (request == nullptr) {
    return UnknownRequest(request_id);
  }
  // A committed output matched a stop id: finish successfully from a ready
  // state (m5.md section 13.4). The reservation releases at terminal
  // emission; no response is emitted by this transition itself.
  RequestEvent event{RequestEventKind::kStopMatched, StopMatchedPayload{}};
  absl::StatusOr<PreparedTransition> prepared = controller_.Prepare(*request, event);
  if (!prepared.ok()) {
    return prepared.status();
  }
  const RequestState from = request->state;
  controller_.Commit(*request, std::move(*prepared));
  return ObserveTransition(*request, from, event.kind, now);
}

absl::Status LifecycleCoordinator::Cancel(RequestId request_id, FinishReason finish,
                                          absl::Status status, ErrorReason reason,
                                          MonotonicTime now) {
  RequestContext* request = registry_.Find(request_id);
  if (request == nullptr) {
    return UnknownRequest(request_id);
  }
  const RequestEventKind kind = finish == FinishReason::kDeadline
                                    ? RequestEventKind::kDeadlineExpired
                                    : RequestEventKind::kCancelRequested;
  RequestEvent event{kind, TerminalOutcomePayload{std::move(status), reason, finish}};
  absl::StatusOr<PreparedTransition> prepared = controller_.Prepare(*request, event);
  if (!prepared.ok()) {
    return prepared.status();
  }
  controller_.StampTerminalTime(*prepared, now);
  const bool immediate = !IsInFlight(request->state);
  std::optional<ResponseReservation> response;
  std::optional<ResourceObservation> release_observation;
  if (immediate) {
    const TerminalOutcomePayload& outcome = std::get<TerminalOutcomePayload>(event.payload);
    auto reserved = responses_.Prepare(
        ResponseEvent{TerminalFrom(*request, finish, outcome.status.code(), reason, now)});
    if (!reserved.ok()) {
      return reserved.status();
    }
    response = std::move(*reserved);
    if (request->reservation.has_value()) {
      absl::StatusOr<ResourceObservation> released = ReleaseReservation(*request, now);
      if (!released.ok()) {
        return released.status();
      }
      release_observation = *released;
    }
  }
  const RequestState from = request->state;
  controller_.Commit(*request, std::move(*prepared));
  if (response.has_value()) {
    responses_.Commit(std::move(*response));
  }
  if (release_observation.has_value()) {
    absl::Status observed_resource = observer_.ObserveResource(*release_observation);
    if (!observed_resource.ok()) {
      return observed_resource;
    }
  }
  return ObserveTransition(*request, from, event.kind, now);
}

absl::Status LifecycleCoordinator::Fail(RequestId request_id, absl::Status status,
                                        ErrorReason reason, MonotonicTime now) {
  RequestContext* request = registry_.Find(request_id);
  if (request == nullptr) {
    return UnknownRequest(request_id);
  }
  RequestEvent event{
      RequestEventKind::kFatalError,
      TerminalOutcomePayload{std::move(status), reason, FinishReason::kExecutorError}};
  absl::StatusOr<PreparedTransition> prepared = controller_.Prepare(*request, event);
  if (!prepared.ok()) {
    return prepared.status();
  }
  controller_.StampTerminalTime(*prepared, now);
  const TerminalOutcomePayload& outcome = std::get<TerminalOutcomePayload>(event.payload);
  auto response = responses_.Prepare(ResponseEvent{
      TerminalFrom(*request, FinishReason::kExecutorError, outcome.status.code(), reason, now)});
  if (!response.ok()) {
    return response.status();
  }
  std::optional<ResourceObservation> release_observation;
  if (request->reservation.has_value()) {
    absl::StatusOr<ResourceObservation> released = ReleaseReservation(*request, now);
    if (!released.ok()) {
      return released.status();
    }
    release_observation = *released;
  }
  const RequestState from = request->state;
  controller_.Commit(*request, std::move(*prepared));
  responses_.Commit(std::move(*response));
  if (release_observation.has_value()) {
    absl::Status observed_resource = observer_.ObserveResource(*release_observation);
    if (!observed_resource.ok()) {
      return observed_resource;
    }
  }
  return ObserveTransition(*request, from, event.kind, now);
}

absl::Status LifecycleCoordinator::Preempt(RequestId request_id, MonotonicTime now) {
  RequestContext* request = registry_.Find(request_id);
  if (request == nullptr) {
    return UnknownRequest(request_id);
  }
  RequestEvent event{RequestEventKind::kPreempt, PreemptPayload{}};
  absl::StatusOr<PreparedTransition> prepared = controller_.Prepare(*request, event);
  if (!prepared.ok()) {
    return prepared.status();
  }
  std::optional<ResourceObservation> release_observation;
  if (request->reservation.has_value()) {
    absl::StatusOr<ResourceObservation> released = ReleaseReservation(*request, now);
    if (!released.ok()) {
      return released.status();
    }
    release_observation = *released;
  }
  const RequestState from = request->state;
  controller_.Commit(*request, std::move(*prepared));
  if (release_observation.has_value()) {
    absl::Status observed_resource = observer_.ObserveResource(*release_observation);
    if (!observed_resource.ok()) {
      return observed_resource;
    }
  }
  return ObserveTransition(*request, from, event.kind, now);
}

absl::Status LifecycleCoordinator::Requeue(RequestId request_id, MonotonicTime now) {
  RequestContext* request = registry_.Find(request_id);
  if (request == nullptr) {
    return UnknownRequest(request_id);
  }
  if (request->epoch.value() == UINT32_MAX) {
    return WithErrorReason(absl::OutOfRangeError("lifecycle.requeue: epoch exhausted"),
                           ErrorReason::kInvariantViolation);
  }
  RequestEvent event{RequestEventKind::kRequeue, RequeuePayload{}};
  absl::StatusOr<PreparedTransition> prepared = controller_.Prepare(*request, event);
  if (!prepared.ok()) {
    return prepared.status();
  }
  const RequestState from = request->state;
  controller_.Commit(*request, std::move(*prepared));
  return ObserveTransition(*request, from, event.kind, now);
}

absl::Status LifecycleCoordinator::EmitTerminal(RequestId request_id, FinishReason reason,
                                                MonotonicTime now) {
  RequestContext* request = registry_.Find(request_id);
  if (request == nullptr) {
    return UnknownRequest(request_id);
  }
  RequestEvent event{RequestEventKind::kTerminalEmitted, TerminalEmittedPayload{reason}};
  absl::StatusOr<PreparedTransition> prepared = controller_.Prepare(*request, event);
  if (!prepared.ok()) {
    return prepared.status();
  }
  controller_.StampTerminalTime(*prepared, now);
  auto response = responses_.Prepare(
      ResponseEvent{TerminalFrom(*request, reason, absl::StatusCode::kOk, std::nullopt, now)});
  if (!response.ok()) {
    return response.status();
  }
  std::optional<ResourceObservation> release_observation;
  if (request->reservation.has_value()) {
    absl::StatusOr<ResourceObservation> released = ReleaseReservation(*request, now);
    if (!released.ok()) {
      return released.status();
    }
    release_observation = *released;
  }
  const RequestState from = request->state;
  controller_.Commit(*request, std::move(*prepared));
  responses_.Commit(std::move(*response));
  if (release_observation.has_value()) {
    absl::Status observed_resource = observer_.ObserveResource(*release_observation);
    if (!observed_resource.ok()) {
      return observed_resource;
    }
  }
  return ObserveTransition(*request, from, event.kind, now);
}

absl::Status LifecycleCoordinator::ObserveTransition(const RequestContext& request,
                                                     RequestState from, RequestEventKind event,
                                                     MonotonicTime now) {
  const absl::StatusCode status =
      request.terminal.has_value() ? request.terminal->status : absl::StatusCode::kOk;
  const ErrorReason reason =
      request.terminal.has_value() && request.terminal->error_reason.has_value()
          ? *request.terminal->error_reason
          : ErrorReason::kNone;
  return observer_.ObserveTransition(TransitionObservation{
      .at = now,
      .request = request.request->id,
      .sequence = request.sequence,
      .epoch = request.epoch,
      .from = from,
      .event = event,
      .to = request.state,
      .status = status,
      .reason = reason,
  });
}

absl::StatusOr<ResourceObservation> LifecycleCoordinator::ReleaseReservation(
    const RequestContext& request, MonotonicTime now) {
  if (!request.reservation.has_value()) {
    return WithErrorReason(
        absl::FailedPreconditionError("lifecycle.release: request has no reservation"),
        ErrorReason::kInvariantViolation);
  }
  const ReservationId id = *request.reservation;
  const std::optional<scheduler::ResourceCost> cost = resources_.LookupCost(id);
  if (!cost.has_value()) {
    return WithErrorReason(absl::InternalError("lifecycle.release: unknown reservation"),
                           ErrorReason::kInvariantViolation);
  }
  absl::Status released = resources_.Release(id);
  if (!released.ok()) {
    return released;
  }
  return ResourceObservation{
      .at = now,
      .action = ResourceAction::kRelease,
      .reservation = id,
      .request = request.request->id,
      .cost = *cost,
      .snapshot = resources_.Snapshot(),
  };
}

}  // namespace inferx
