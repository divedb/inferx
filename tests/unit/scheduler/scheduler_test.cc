#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/scheduler/admission_controller.h"
#include "inferx/scheduler/batch_planner.h"
#include "inferx/scheduler/fcfs_policy.h"
#include "inferx/scheduler/resource_accountant.h"
#include "inferx/scheduler/scheduling_snapshot.h"
#include "inferx/scheduler/step_plan.h"
#include "inferx/scheduler/step_plan_validator.h"

namespace inferx::scheduler {
namespace {

RequestSchedulingView View(uint64_t id, RequestState state, uint64_t arrival, uint32_t prompt,
                           uint32_t output, std::optional<ReservationId> reservation) {
  return RequestSchedulingView{
      .request = RequestId(id),
      .sequence = SequenceId(id + 100),
      .epoch = RequestEpoch(0),
      .model = ModelId(0),
      .state = state,
      .arrival_ordinal = arrival,
      .arrival_time = MonotonicTime(Nanoseconds(static_cast<int64_t>(arrival))),
      .prompt_tokens = TokenCount(prompt),
      .computed_tokens = TokenCount(state == RequestState::kDecodeReady ? prompt : 0),
      .committed_output_tokens = TokenCount(output),
      .max_output_tokens = TokenCount(8),
      .deadline = std::nullopt,
      .reservation = reservation,
  };
}

SchedulingSnapshot Snapshot(std::span<const RequestSchedulingView> requests, uint32_t sequences = 8,
                            uint32_t tokens = 32) {
  return SchedulingSnapshot{
      .now = MonotonicTime(Nanoseconds(20)),
      .proposed_step = StepId(1),
      .requests = requests,
      .limits = SchedulingLimits{SequenceCount(sequences), TokenCount(tokens)},
      .resources =
          ResourceSnapshot{SequenceCount(8), SequenceCount(0), KvTokenCount(1024), KvTokenCount(0)},
  };
}

TEST(ResourceAccountantTest, CommitReleaseAndRollbackAreTransactional) {
  ResourceAccountant accountant(SequenceCount(2), KvTokenCount(20));
  auto transaction =
      accountant.BeginReservation(RequestId(1), ResourceCost{SequenceCount(1), TokenCount(7)});
  ASSERT_TRUE(transaction.ok()) << transaction.status();
  EXPECT_EQ(transaction->id(), ReservationId(1));
  EXPECT_EQ(accountant.Snapshot().sequences_used, SequenceCount(0));
  EXPECT_TRUE(accountant.has_tentative());

  const ReservationId committed = transaction->Commit();
  EXPECT_EQ(committed, ReservationId(1));
  EXPECT_EQ(accountant.Snapshot().sequences_used, SequenceCount(1));
  EXPECT_EQ(accountant.Snapshot().kv_tokens_used, KvTokenCount(7));
  EXPECT_TRUE(accountant.HasReservation(committed));
  EXPECT_TRUE(accountant.Validate().ok());

  {
    auto rolled_back =
        accountant.BeginReservation(RequestId(2), ResourceCost{SequenceCount(1), TokenCount(5)});
    ASSERT_TRUE(rolled_back.ok()) << rolled_back.status();
    EXPECT_EQ(rolled_back->id(), ReservationId(2));
  }
  EXPECT_FALSE(accountant.has_tentative());
  auto reused =
      accountant.BeginReservation(RequestId(2), ResourceCost{SequenceCount(1), TokenCount(5)});
  ASSERT_TRUE(reused.ok()) << reused.status();
  EXPECT_EQ(reused->Commit(), ReservationId(2));

  EXPECT_TRUE(accountant.Release(ReservationId(1)).ok());
  EXPECT_TRUE(accountant.Release(ReservationId(2)).ok());
  EXPECT_EQ(accountant.Snapshot().sequences_used, SequenceCount(0));
  EXPECT_EQ(accountant.Snapshot().kv_tokens_used, KvTokenCount(0));
  EXPECT_TRUE(accountant.Validate().ok());
}

TEST(ResourceAccountantTest, CapacityDuplicateAndUnknownReleaseDoNotMutate) {
  ResourceAccountant accountant(SequenceCount(1), KvTokenCount(4));
  auto first =
      accountant.BeginReservation(RequestId(1), ResourceCost{SequenceCount(1), TokenCount(4)});
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(first->Commit(), ReservationId(1));
  const ResourceSnapshot full = accountant.Snapshot();

  auto duplicate =
      accountant.BeginReservation(RequestId(1), ResourceCost{SequenceCount(1), TokenCount(1)});
  EXPECT_EQ(duplicate.status().code(), absl::StatusCode::kAlreadyExists);
  auto exhausted =
      accountant.BeginReservation(RequestId(2), ResourceCost{SequenceCount(1), TokenCount(1)});
  EXPECT_EQ(exhausted.status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(accountant.Snapshot(), full);
  EXPECT_EQ(accountant.Release(ReservationId(99)).code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(accountant.Snapshot(), full);
}

TEST(FcfsPolicyTest, SelectsEligibleUnexpiredRequestsInStableOrder) {
  std::array<RequestSchedulingView, 4> views{
      View(1, RequestState::kQueued, 1, 2, 0, std::nullopt),
      View(2, RequestState::kPrefilling, 2, 2, 0, ReservationId(1)),
      View(3, RequestState::kPrefillReady, 3, 2, 0, ReservationId(2)),
      View(4, RequestState::kDecodeReady, 4, 2, 1, ReservationId(3)),
  };
  views[2].deadline = MonotonicTime(Nanoseconds(20));
  std::array<size_t, 4> output{99, 99, 99, 99};
  auto count = FcfsPolicy().Select(Snapshot(views), output);
  ASSERT_TRUE(count.ok()) << count.status();
  EXPECT_EQ(*count, 2U);
  EXPECT_EQ(output[0], 0U);
  EXPECT_EQ(output[1], 3U);
}

TEST(FcfsPolicyTest, RejectsUnorderedInputAndSmallScratchWithoutPartialWrite) {
  std::array<RequestSchedulingView, 2> views{
      View(1, RequestState::kQueued, 2, 2, 0, std::nullopt),
      View(2, RequestState::kQueued, 1, 2, 0, std::nullopt),
  };
  std::array<size_t, 2> output{77, 88};
  auto unordered = FcfsPolicy().Select(Snapshot(views), output);
  EXPECT_EQ(unordered.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(output, (std::array<size_t, 2>{77, 88}));

  std::swap(views[0], views[1]);
  auto too_small = FcfsPolicy().Select(Snapshot(views), std::span<size_t>(output).first(1));
  EXPECT_EQ(too_small.status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(output, (std::array<size_t, 2>{77, 88}));
}

TEST(AdmissionControllerTest, DistinguishesReservableTemporaryAndImpossible) {
  const RequestSchedulingView request = View(1, RequestState::kQueued, 1, 3, 0, std::nullopt);
  AdmissionController admission;
  auto available = admission.Evaluate(request, ResourceSnapshot{SequenceCount(2), SequenceCount(0),
                                                                KvTokenCount(32), KvTokenCount(0)});
  EXPECT_EQ(available.kind, AdmissionKind::kReservable);
  EXPECT_EQ(available.cost, (ResourceCost{SequenceCount(1), TokenCount(11)}));

  auto blocked = admission.Evaluate(request, ResourceSnapshot{SequenceCount(2), SequenceCount(2),
                                                              KvTokenCount(32), KvTokenCount(0)});
  EXPECT_EQ(blocked.kind, AdmissionKind::kTemporarilyBlocked);
  EXPECT_TRUE(blocked.rejection.ok());

  auto impossible = admission.Evaluate(
      request,
      ResourceSnapshot{SequenceCount(2), SequenceCount(0), KvTokenCount(10), KvTokenCount(0)});
  EXPECT_EQ(impossible.kind, AdmissionKind::kNeverFits);
  EXPECT_EQ(impossible.rejection.code(), absl::StatusCode::kResourceExhausted);
}

TEST(StepPlanPoolTest, ExhaustionMoveAndGenerationReuseAreExact) {
  auto pool = StepPlanPool::Create(1, SequenceCount(2));
  ASSERT_TRUE(pool.ok()) << pool.status();
  {
    auto first = (*pool)->Acquire(StepId(1), ModelId(0), MonotonicTime{});
    ASSERT_TRUE(first.ok()) << first.status();
    EXPECT_EQ(first->plan().buffer_generation, PlanBufferGeneration(1));
    EXPECT_EQ((*pool)->available(), 0U);
    auto exhausted = (*pool)->Acquire(StepId(2), ModelId(0), MonotonicTime{});
    EXPECT_EQ(exhausted.status().code(), absl::StatusCode::kResourceExhausted);

    StepPlanLease moved(std::move(*first));
    EXPECT_FALSE(first->valid());
    EXPECT_TRUE(moved.valid());
  }
  EXPECT_EQ((*pool)->available(), 1U);

  auto second = (*pool)->Acquire(StepId(2), ModelId(0), MonotonicTime{});
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(second->plan().buffer_generation, PlanBufferGeneration(2));
  EXPECT_TRUE((*pool)->Validate().ok());
}

TEST(BatchPlannerTest, BuildsAndValidatesPrefillAndDecodeRanges) {
  std::array<RequestSchedulingView, 2> views{
      View(1, RequestState::kPrefillReady, 1, 3, 0, ReservationId(10)),
      View(2, RequestState::kDecodeReady, 2, 2, 1, ReservationId(11)),
  };
  const SchedulingSnapshot snapshot = Snapshot(views, 2, 8);
  auto pool = StepPlanPool::Create(1, SequenceCount(2));
  ASSERT_TRUE(pool.ok());
  auto lease = (*pool)->Acquire(snapshot.proposed_step, ModelId(0), snapshot.now);
  ASSERT_TRUE(lease.ok());
  const std::array<size_t, 2> selected{0, 1};
  ASSERT_TRUE(BatchPlanner().Build(snapshot, selected, *lease).ok());

  const StepPlan& plan = lease->plan();
  ASSERT_EQ(plan.sequences.size(), 2U);
  EXPECT_EQ(plan.resources.num_tokens, TokenCount(4));
  EXPECT_EQ(plan.sequences[0].input_tokens.begin, TokenOffset(0));
  EXPECT_EQ(plan.sequences[0].input_tokens.end, TokenOffset(3));
  EXPECT_EQ(plan.sequences[1].input_tokens.begin, TokenOffset(2));
  EXPECT_EQ(plan.sequences[1].input_tokens.end, TokenOffset(3));
  EXPECT_EQ(plan.sequences[1].kv_read.logical_tokens.end, TokenOffset(2));
  EXPECT_TRUE(StepPlanValidator().Validate(plan, snapshot).ok());
}

TEST(BatchPlannerTest, HeadOfLineBudgetBlockDoesNotBypassEarlierRequest) {
  std::array<RequestSchedulingView, 2> views{
      View(1, RequestState::kPrefillReady, 1, 5, 0, ReservationId(10)),
      View(2, RequestState::kDecodeReady, 2, 2, 1, ReservationId(11)),
  };
  const SchedulingSnapshot snapshot = Snapshot(views, 2, 4);
  auto pool = StepPlanPool::Create(1, SequenceCount(2));
  ASSERT_TRUE(pool.ok());
  auto lease = (*pool)->Acquire(snapshot.proposed_step, ModelId(0), snapshot.now);
  ASSERT_TRUE(lease.ok());
  const std::array<size_t, 2> selected{0, 1};
  EXPECT_TRUE(BatchPlanner().Build(snapshot, selected, *lease).ok());
  EXPECT_TRUE(lease->plan().sequences.empty());
  EXPECT_EQ(StepPlanValidator().Validate(lease->plan(), snapshot).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(StepPlanValidatorTest, RejectsIdentityAndResourceMismatches) {
  std::array<RequestSchedulingView, 1> views{
      View(1, RequestState::kPrefillReady, 1, 3, 0, ReservationId(10)),
  };
  const SchedulingSnapshot snapshot = Snapshot(views, 1, 8);
  auto pool = StepPlanPool::Create(1, SequenceCount(1));
  ASSERT_TRUE(pool.ok());
  auto lease = (*pool)->Acquire(snapshot.proposed_step, ModelId(0), snapshot.now);
  ASSERT_TRUE(lease.ok());
  const std::array<size_t, 1> selected{0};
  ASSERT_TRUE(BatchPlanner().Build(snapshot, selected, *lease).ok());

  StepPlan wrong_step = lease->plan();
  wrong_step.id = StepId(9);
  EXPECT_EQ(StepPlanValidator().Validate(wrong_step, snapshot).code(),
            absl::StatusCode::kInvalidArgument);
  StepPlan wrong_total = lease->plan();
  wrong_total.resources.num_tokens = TokenCount(2);
  EXPECT_EQ(StepPlanValidator().Validate(wrong_total, snapshot).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx::scheduler
