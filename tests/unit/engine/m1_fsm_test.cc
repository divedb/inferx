// Request FSM / controller / registry tests (m1.md sections 10-11, 17.1).
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "inferx/api/generate_request.h"
#include "inferx/base/token.h"
#include "inferx/engine/request_controller.h"
#include "inferx/engine/request_registry.h"
#include "inferx/engine/request_state_machine.h"
#include "inferx/lifecycle/request_state.h"

namespace {

using inferx::ExecutionCompletion;
using inferx::ExecutionTicket;
using inferx::FinishReason;
using inferx::GenerateRequest;
using inferx::GenerationLimits;
using inferx::PreparedTransition;
using inferx::RequestContext;
using inferx::RequestController;
using inferx::RequestEvent;
using inferx::RequestEventKind;
using inferx::RequestId;
using inferx::RequestRegistry;
using inferx::RequestState;
using inferx::SubmitPayload;
using inferx::TokenCount;
using inferx::TokenId;
using inferx::TokenOffset;
using inferx::TokenRange;

std::unique_ptr<RequestContext> AdmittedContext(RequestId id, uint32_t max_output = 5) {
  auto context = std::make_unique<RequestContext>();
  const GenerateRequest request = GenerateRequest{
      .id = id,
      .model = inferx::ModelId(0),
      .input = std::vector<TokenId>{TokenId(1), TokenId(2), TokenId(3)},
      .generation = GenerationLimits{.max_output_tokens = TokenCount(max_output)},
      .deadline = std::nullopt,
  };
  context->request = std::make_unique<const GenerateRequest>(request);
  context->prompt_tokens = {TokenId(1), TokenId(2), TokenId(3)};
  context->state = RequestState::kQueued;
  return context;
}

RequestEvent Marker(RequestEventKind kind) {
  switch (kind) {
    case RequestEventKind::kStartTokenization:
      return {kind, inferx::StartTokenizationPayload{}};
    case RequestEventKind::kBeginReservation:
      return {kind, inferx::BeginReservationPayload{}};
    case RequestEventKind::kReservationDeferred:
      return {kind, inferx::ReservationDeferredPayload{}};
    case RequestEventKind::kPreempt:
      return {kind, inferx::PreemptPayload{}};
    case RequestEventKind::kRequeue:
      return {kind, inferx::RequeuePayload{}};
    default:
      return {kind, inferx::TerminalEmittedPayload{}};
  }
}

RequestEvent CancelEvent(absl::StatusCode code = absl::StatusCode::kCancelled) {
  return {RequestEventKind::kCancelRequested,
          inferx::TerminalOutcomePayload{absl::Status(code, "test cancel"),
                                         inferx::ErrorReason::kExplicitCancellation,
                                         FinishReason::kCancelled}};
}

ExecutionTicket TicketFor(const RequestContext& context) {
  const inferx::StepId step = context.in_flight_step.value_or(inferx::StepId(0));
  const inferx::ExecutionTicketId ticket =
      context.in_flight_ticket.value_or(inferx::ExecutionTicketId(0));
  return ExecutionTicket{ticket, step, 1};
}

ExecutionCompletion CompletionFor(const RequestContext& context,
                                  std::optional<inferx::RequestEpoch> epoch = std::nullopt,
                                  const absl::Status& status = absl::OkStatus()) {
  const TokenRange range =
      context.in_flight_range.value_or(TokenRange{TokenOffset(0), TokenOffset(1)});
  return ExecutionCompletion{
      .ticket = context.in_flight_ticket.value_or(inferx::ExecutionTicketId(0)),
      .step = context.in_flight_step.value_or(inferx::StepId(0)),
      .item_ordinal = 0,
      .item_count = 1,
      .request = context.request->id,
      .sequence = context.sequence,
      .epoch = epoch.value_or(context.epoch),
      .kind = context.in_flight_work.value_or(inferx::WorkKind::kDecode),
      .scheduled_tokens = range,
      .status = status,
      .error_reason =
          status.ok() ? inferx::ErrorReason::kNone : inferx::ErrorReason::kExecutorFailure,
      .output_token =
          status.ok() ? std::optional<inferx::TokenId>(inferx::TokenId(42)) : std::nullopt,
  };
}

class Controlled {
 public:
  absl::Status Apply(RequestContext& context, const RequestEvent& event) {
    absl::StatusOr<PreparedTransition> prepared = controller_.Prepare(context, event);
    if (!prepared.ok()) {
      return prepared.status();
    }
    controller_.Commit(context, std::move(*prepared));
    return absl::OkStatus();
  }

 private:
  RequestController controller_;
};

TEST(FsmTest, HappyPathPrefillThenDecodesThenLength) {
  Controlled fsm;
  auto context = AdmittedContext(RequestId(1), 2);
  context->state = RequestState::kPrefillReady;
  context->reservation = inferx::ReservationId(1);

  // Submit + complete prefill: commits first synthetic token.
  ASSERT_TRUE(
      fsm.Apply(*context, RequestEvent{RequestEventKind::kSubmitPrefill,
                                       SubmitPayload{.step = inferx::StepId(1),
                                                     .ticket = inferx::ExecutionTicketId(1),
                                                     .epoch = context->epoch,
                                                     .work = inferx::WorkKind::kPrefill,
                                                     .scheduled_range =
                                                         TokenRange{TokenOffset(0), TokenOffset(3)},
                                                     .item_count = 1}})
          .ok());
  EXPECT_EQ(context->state, RequestState::kPrefilling);
  EXPECT_EQ(context->num_scheduled_tokens, 3u);

  ASSERT_TRUE(
      fsm.Apply(*context, RequestEvent{RequestEventKind::kPrefillCompleted,
                                       inferx::PrefillCompletedPayload{CompletionFor(*context)}})
          .ok());
  EXPECT_EQ(context->state, RequestState::kDecodeReady);
  EXPECT_EQ(context->num_computed_tokens, 3u);
  EXPECT_EQ(context->num_committed_output_tokens, 1u);
  EXPECT_EQ(context->num_scheduled_tokens, 0u);

  // Decode one token; output budget of 2 is then exhausted.
  ASSERT_TRUE(
      fsm.Apply(*context, RequestEvent{RequestEventKind::kSubmitDecode,
                                       SubmitPayload{.step = inferx::StepId(2),
                                                     .ticket = inferx::ExecutionTicketId(2),
                                                     .epoch = context->epoch,
                                                     .work = inferx::WorkKind::kDecode,
                                                     .scheduled_range =
                                                         TokenRange{TokenOffset(3), TokenOffset(4)},
                                                     .item_count = 1}})
          .ok());
  ASSERT_TRUE(
      fsm.Apply(*context, RequestEvent{RequestEventKind::kDecodeCompleted,
                                       inferx::DecodeCompletedPayload{CompletionFor(*context)}})
          .ok());
  EXPECT_EQ(context->state, RequestState::kFinishing);

  ASSERT_TRUE(fsm.Apply(*context, Marker(RequestEventKind::kTerminalEmitted)).ok());
  EXPECT_EQ(context->state, RequestState::kFinished);
  const inferx::TerminalResponse terminal = context->terminal.value_or(inferx::TerminalResponse{});
  EXPECT_EQ(terminal.reason, FinishReason::kLength);
  EXPECT_EQ(terminal.prompt_tokens, TokenCount(3));
  EXPECT_EQ(terminal.output_tokens, TokenCount(2));
  EXPECT_TRUE(context->terminal_emitted);
  EXPECT_FALSE(context->reservation.has_value());  // released at terminal
}

TEST(FsmTest, TerminalStatesAreAbsorbing) {
  Controlled fsm;
  for (const RequestState terminal :
       {RequestState::kFinished, RequestState::kCancelled, RequestState::kFailed}) {
    auto context = AdmittedContext(RequestId(1));
    context->state = terminal;
    context->terminal_emitted = true;
    const absl::Status rejected = fsm.Apply(*context, Marker(RequestEventKind::kPreempt));
    ASSERT_FALSE(rejected.ok());
    EXPECT_EQ(rejected.code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_EQ(context->state, terminal);
  }
}

TEST(FsmTest, CancellationFromEveryNonterminalNonInFlightState) {
  for (const RequestState state :
       {RequestState::kReceived, RequestState::kTokenizing, RequestState::kQueued,
        RequestState::kReserving, RequestState::kPrefillReady, RequestState::kDecodeReady,
        RequestState::kPreempted}) {
    Controlled fsm;
    auto context = AdmittedContext(RequestId(1));
    context->state = state;
    ASSERT_TRUE(fsm.Apply(*context, CancelEvent()).ok()) << inferx::ToString(state);
    EXPECT_EQ(context->state, RequestState::kCancelled) << inferx::ToString(state);
    const inferx::TerminalResponse terminal =
        context->terminal.value_or(inferx::TerminalResponse{});
    EXPECT_EQ(terminal.reason, FinishReason::kCancelled);
    EXPECT_EQ(terminal.status, absl::StatusCode::kCancelled);
  }
}

TEST(FsmTest, CancelInFlightDrainsThroughCancelling) {
  Controlled fsm;
  auto context = AdmittedContext(RequestId(1));
  context->state = RequestState::kDecoding;
  context->in_flight_step = inferx::StepId(7);
  context->in_flight_ticket = inferx::ExecutionTicketId(7);
  context->submitted_ticket = TicketFor(*context);
  context->in_flight_work = inferx::WorkKind::kDecode;
  context->in_flight_range = TokenRange{TokenOffset(3), TokenOffset(4)};
  context->num_scheduled_tokens = 1;
  context->reservation = inferx::ReservationId(1);

  ASSERT_TRUE(fsm.Apply(*context, CancelEvent()).ok());
  EXPECT_EQ(context->state, RequestState::kCancelling);

  // Double cancel while cancelling is illegal (already draining).
  EXPECT_FALSE(fsm.Apply(*context, CancelEvent()).ok());

  ASSERT_TRUE(
      fsm.Apply(*context, RequestEvent{RequestEventKind::kInFlightDrained,
                                       inferx::InFlightDrainedPayload{CompletionFor(*context)}})
          .ok());
  EXPECT_EQ(context->state, RequestState::kCancelled);
  EXPECT_EQ(context->num_committed_output_tokens, 0U);
  EXPECT_FALSE(context->in_flight_ticket.has_value());
  const std::optional<inferx::TerminalResponse> terminal = context->terminal;
  ASSERT_TRUE(terminal.has_value());
  const inferx::TerminalResponse terminal_response = terminal.value_or(inferx::TerminalResponse{});
  EXPECT_EQ(terminal_response.reason, FinishReason::kCancelled);
}

TEST(FsmTest, StaleCompletionRejectedWithoutSideEffects) {
  Controlled fsm;
  auto context = AdmittedContext(RequestId(1));
  context->state = RequestState::kDecoding;
  context->in_flight_step = inferx::StepId(7);
  context->in_flight_ticket = inferx::ExecutionTicketId(7);
  context->submitted_ticket = TicketFor(*context);
  context->in_flight_work = inferx::WorkKind::kDecode;
  context->in_flight_range = TokenRange{TokenOffset(3), TokenOffset(4)};
  context->num_scheduled_tokens = 1;

  // Wrong epoch (stale after preemption).
  const absl::Status rejected =
      fsm.Apply(*context, RequestEvent{RequestEventKind::kDecodeCompleted,
                                       inferx::DecodeCompletedPayload{
                                           CompletionFor(*context, inferx::RequestEpoch(99))}});
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(context->state, RequestState::kDecoding);
  EXPECT_EQ(context->num_committed_output_tokens, 0u);
}

TEST(FsmTest, PreemptThenRequeueIncrementsEpoch) {
  Controlled fsm;
  auto context = AdmittedContext(RequestId(1));
  context->state = RequestState::kDecodeReady;
  context->reservation = inferx::ReservationId(4);

  ASSERT_TRUE(fsm.Apply(*context, Marker(RequestEventKind::kPreempt)).ok());
  EXPECT_EQ(context->state, RequestState::kPreempted);
  EXPECT_FALSE(context->reservation.has_value());

  const inferx::RequestEpoch before = context->epoch;
  ASSERT_TRUE(fsm.Apply(*context, Marker(RequestEventKind::kRequeue)).ok());
  EXPECT_EQ(context->state, RequestState::kQueued);
  EXPECT_EQ(context->epoch.value(), before.value() + 1);
}

TEST(FsmTest, ReservationGrantedResolvesByComputedTokens) {
  Controlled fsm;
  auto fresh = AdmittedContext(RequestId(1));
  fresh->state = RequestState::kReserving;
  ASSERT_TRUE(fsm.Apply(*fresh, RequestEvent{RequestEventKind::kReservationGranted,
                                             inferx::ReservationGrantedPayload{
                                                 inferx::ReservationId(9),
                                                 inferx::scheduler::ResourceCost{
                                                     inferx::SequenceCount(1), TokenCount(8)}}})
                  .ok());
  EXPECT_EQ(fresh->state, RequestState::kPrefillReady);

  auto resumed = AdmittedContext(RequestId(2));
  resumed->state = RequestState::kReserving;
  resumed->num_computed_tokens = 3;
  ASSERT_TRUE(fsm.Apply(*resumed, RequestEvent{RequestEventKind::kReservationGranted,
                                               inferx::ReservationGrantedPayload{
                                                   inferx::ReservationId(9),
                                                   inferx::scheduler::ResourceCost{
                                                       inferx::SequenceCount(1), TokenCount(8)}}})
                  .ok());
  EXPECT_EQ(resumed->state, RequestState::kDecodeReady);
}

TEST(RegistryTest, DuplicateInsertAndDeterministicOrder) {
  RequestRegistry registry;
  auto first = AdmittedContext(RequestId(10));
  auto second = AdmittedContext(RequestId(20));
  ASSERT_TRUE(registry.Insert(std::move(first)).ok());
  const absl::Status duplicate = registry.Insert(AdmittedContext(RequestId(10)));
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.code(), absl::StatusCode::kAlreadyExists);
  ASSERT_TRUE(registry.Insert(std::move(second)).ok());

  std::vector<const RequestContext*> order;
  registry.AppendArrivalOrder(order);
  ASSERT_EQ(order.size(), 2u);
  EXPECT_EQ(order[0]->request->id, RequestId(10));
  EXPECT_EQ(order[1]->request->id, RequestId(20));
  EXPECT_EQ(order[0]->sequence, inferx::SequenceId(1));
  EXPECT_EQ(order[1]->sequence, inferx::SequenceId(2));
}

TEST(RegistryTest, EraseRequiresTerminalCleanState) {
  RequestRegistry registry;
  ASSERT_TRUE(registry.Insert(AdmittedContext(RequestId(1))).ok());
  EXPECT_FALSE(registry.EraseTerminal(RequestId(1)).ok());  // no terminal yet

  RequestContext* context = registry.Find(RequestId(1));
  ASSERT_NE(context, nullptr);
  context->state = RequestState::kCancelled;
  context->terminal_emitted = true;
  ASSERT_TRUE(registry.EraseTerminal(RequestId(1)).ok());
  EXPECT_FALSE(registry.Contains(RequestId(1)));
}

TEST(TableTest, NoDuplicateCellsInRuleArray) {
  const auto rules = inferx::TransitionRules();
  for (size_t i = 0; i < rules.size; ++i) {
    for (size_t j = i + 1; j < rules.size; ++j) {
      ASSERT_FALSE(rules.data[i].current == rules.data[j].current &&
                   rules.data[i].event == rules.data[j].event)
          << "duplicate cell at " << i << " and " << j;
    }
  }
}

TEST(NamesTest, StateAndEventNamesRoundTrip) {
  for (uint32_t value = 0; value <= 13; ++value) {
    const auto state = static_cast<RequestState>(value);
    const auto parsed = inferx::RequestStateFromName(inferx::ToString(state));
    ASSERT_TRUE(parsed.has_value()) << inferx::ToString(state);
    EXPECT_EQ(parsed.value_or(RequestState::kFailed), state);
  }
  EXPECT_FALSE(inferx::RequestStateFromName("PrefillReady").has_value());
  EXPECT_FALSE(inferx::RequestStateFromName("").has_value());
}

}  // namespace
