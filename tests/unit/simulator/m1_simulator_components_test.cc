#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/scheduler/batch_planner.h"
#include "inferx/scheduler/scheduling_snapshot.h"
#include "inferx/scheduler/step_plan.h"
#include "inferx/simulator/fake_executor.h"
#include "inferx/simulator/latency_model.h"
#include "inferx/simulator/workload.h"

namespace inferx::simulator {
namespace {

struct PlanFixture {
  std::unique_ptr<scheduler::StepPlanPool> pool;
  scheduler::StepPlanLease lease;
};

PlanFixture PrefillPlan(StepId step = StepId(1), RequestId request = RequestId(7),
                        MonotonicTime now = MonotonicTime{}) {
  auto pool = scheduler::StepPlanPool::Create(1, SequenceCount(1));
  EXPECT_TRUE(pool.ok());
  std::unique_ptr<scheduler::StepPlanPool> owned_pool = std::move(*pool);
  auto lease = owned_pool->Acquire(step, ModelId(0), now);
  EXPECT_TRUE(lease.ok());
  const std::array<scheduler::RequestSchedulingView, 1> views{
      scheduler::RequestSchedulingView{
          .request = request,
          .sequence = SequenceId(request.value() + 100),
          .epoch = RequestEpoch(0),
          .model = ModelId(0),
          .state = RequestState::kPrefillReady,
          .arrival_ordinal = request.value(),
          .arrival_time = now,
          .prompt_tokens = TokenCount(3),
          .computed_tokens = TokenCount(0),
          .committed_output_tokens = TokenCount(0),
          .max_output_tokens = TokenCount(2),
          .deadline = std::nullopt,
          .reservation = ReservationId(request.value()),
      },
  };
  const scheduler::SchedulingSnapshot snapshot{
      .now = now,
      .proposed_step = step,
      .requests = views,
      .limits = scheduler::SchedulingLimits{SequenceCount(1), TokenCount(8)},
      .resources = scheduler::ResourceSnapshot{SequenceCount(1), SequenceCount(1), KvTokenCount(5),
                                               KvTokenCount(5)},
  };
  const std::array<size_t, 1> selected{0};
  EXPECT_TRUE(scheduler::BatchPlanner().Build(snapshot, selected, *lease).ok());
  return PlanFixture{std::move(owned_pool), std::move(*lease)};
}

TEST(LatencyModelTest, UsesCheckedSchemaFormula) {
  PlanFixture fixture = PrefillPlan();
  LatencyModel latency(LatencyModelParameters{
      .base = Nanoseconds(100),
      .prefill_per_token = Nanoseconds(10),
      .decode_per_sequence = Nanoseconds(20),
  });
  auto duration = latency.Duration(fixture.lease.plan());
  ASSERT_TRUE(duration.ok()) << duration.status();
  EXPECT_EQ(*duration, Nanoseconds(130));
  auto completion = latency.CompletionTime(fixture.lease.plan());
  ASSERT_TRUE(completion.ok());
  EXPECT_EQ(*completion, MonotonicTime(Nanoseconds(130)));
}

TEST(FakeExecutorTest, TicketDeliveryAndAcknowledgementOwnPlanLease) {
  PlanFixture fixture = PrefillPlan();
  auto executor = FakeExecutor::Create(
      LatencyModel(LatencyModelParameters{
          .base = Nanoseconds(100),
          .prefill_per_token = Nanoseconds(10),
          .decode_per_sequence = Nanoseconds(20),
      }),
      1, 2);
  ASSERT_TRUE(executor.ok());
  auto ticket = (*executor)->Submit(std::move(fixture.lease));
  ASSERT_TRUE(ticket.ok()) << ticket.status();
  EXPECT_EQ(ticket->id, ExecutionTicketId(1));
  EXPECT_EQ(fixture.pool->available(), 0U);
  EXPECT_EQ((*executor)->Acknowledge(ticket->id).code(), absl::StatusCode::kFailedPrecondition);

  std::array<ExecutionCompletion, 1> completions;
  auto early = (*executor)->CompleteReady(MonotonicTime(Nanoseconds(129)), completions);
  ASSERT_TRUE(early.ok());
  EXPECT_EQ(*early, 0U);
  auto ready = (*executor)->CompleteReady(MonotonicTime(Nanoseconds(130)), completions);
  ASSERT_TRUE(ready.ok());
  ASSERT_EQ(*ready, 1U);
  EXPECT_TRUE(completions[0].status.ok());
  EXPECT_EQ(completions[0].output_token, FakeToken(RequestId(7), TokenOffset(0)));
  EXPECT_EQ((*executor)->delivered_tickets(), 1U);
  EXPECT_TRUE((*executor)->Acknowledge(ticket->id).ok());
  EXPECT_EQ(fixture.pool->available(), 1U);
  EXPECT_EQ((*executor)->Acknowledge(ticket->id).code(), absl::StatusCode::kNotFound);
}

TEST(FakeExecutorTest, RequestFailureRuleIsConsumedExactlyOnce) {
  PlanFixture fixture = PrefillPlan();
  auto executor = FakeExecutor::Create(
      LatencyModel(LatencyModelParameters{
          .base = Nanoseconds(1),
          .prefill_per_token = Nanoseconds(0),
          .decode_per_sequence = Nanoseconds(0),
      }),
      1, 1);
  ASSERT_TRUE(executor.ok());
  ASSERT_TRUE((*executor)
                  ->AddFailureRule(FailureRule{
                      .target_kind = FailureTargetKind::kRequest,
                      .target_id = 7,
                      .status = absl::InternalError("injected"),
                      .reason = ErrorReason::kExecutorFailure,
                      .persistent = false,
                  })
                  .ok());
  auto ticket = (*executor)->Submit(std::move(fixture.lease));
  ASSERT_TRUE(ticket.ok());
  std::array<ExecutionCompletion, 1> completions;
  auto ready = (*executor)->CompleteReady(MonotonicTime(Nanoseconds(1)), completions);
  ASSERT_TRUE(ready.ok());
  ASSERT_EQ(*ready, 1U);
  EXPECT_EQ(completions[0].status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(completions[0].error_reason, ErrorReason::kExecutorFailure);
  EXPECT_FALSE(completions[0].output_token.has_value());
  EXPECT_EQ((*executor)->unconsumed_failure_rules(), 0U);
}

TEST(WorkloadTest, StrictParserAndCanonicalWriterRoundTripFields) {
  const std::string text =
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":7,\"model_id\":0,\"token_ids\":[1,-2,3],"
      "\"max_output_tokens\":2,\"deadline_ns\":null,"
      "\"finish_after_tokens\":2,\"priority\":0,\"tenant_scope\":0}\n"
      "{\"schema_version\":1,\"event_type\":\"cancel\",\"at_ns\":10,"
      "\"request_id\":7}\n";
  auto events = ParseWorkloadJsonLines(text);
  ASSERT_TRUE(events.ok()) << events.status();
  ASSERT_EQ(events->size(), 2U);
  EXPECT_EQ(events->at(0).type, WorkloadEventType::kSubmit);
  const auto& submit = std::get<SubmitWorkloadEvent>(events->at(0).payload);
  EXPECT_EQ(submit.request.id, RequestId(7));
  EXPECT_EQ(submit.finish_after_tokens, TokenCount(2));
  EXPECT_EQ(CanonicalWorkloadEvent(events->at(0)), text.substr(0, text.find('\n')));
}

TEST(WorkloadTest, RejectsUnknownDuplicateUnsortedAndInvalidReferences) {
  auto unknown = ParseWorkloadJsonLines(
      "{\"schema_version\":1,\"event_type\":\"cancel\",\"at_ns\":0,"
      "\"request_id\":1,\"extra\":2}");
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  auto duplicate = ParseWorkloadJsonLines(
      "{\"schema_version\":1,\"event_type\":\"cancel\",\"at_ns\":0,"
      "\"request_id\":1,\"request_id\":2}");
  EXPECT_EQ(duplicate.status().code(), absl::StatusCode::kInvalidArgument);
  auto unsorted = ParseWorkloadJsonLines(
      "{\"schema_version\":1,\"event_type\":\"cancel\",\"at_ns\":2,"
      "\"request_id\":1}\n"
      "{\"schema_version\":1,\"event_type\":\"cancel\",\"at_ns\":1,"
      "\"request_id\":1}");
  EXPECT_EQ(unsorted.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx::simulator
