#include "inferx/simulator/engine_simulator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/api/generate_request.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"
#include "inferx/config/engine_config.h"
#include "inferx/engine/event_sink.h"
#include "inferx/engine/lifecycle_coordinator.h"
#include "inferx/engine/request_context.h"
#include "inferx/engine/request_registry.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/scheduler/admission_controller.h"
#include "inferx/scheduler/batch_planner.h"
#include "inferx/scheduler/fcfs_policy.h"
#include "inferx/scheduler/resource_accountant.h"
#include "inferx/scheduler/scheduling_snapshot.h"
#include "inferx/scheduler/step_plan.h"
#include "inferx/scheduler/step_plan_validator.h"
#include "inferx/simulator/fake_executor.h"
#include "inferx/simulator/invariant_checker.h"
#include "inferx/simulator/latency_model.h"

namespace inferx::simulator {
namespace {

absl::Status SimulationError(absl::StatusCode code, absl::string_view detail, ErrorReason reason) {
  return WithErrorReason(absl::Status(code, absl::StrCat("simulator: ", detail)), reason);
}

int EventClass(WorkloadEventType type) {
  switch (type) {
    case WorkloadEventType::kShutdown:
    case WorkloadEventType::kCancel:
      return 1;
    case WorkloadEventType::kInjectExecutorFailure:
      return 2;
    case WorkloadEventType::kSubmit:
    case WorkloadEventType::kPreempt:
    case WorkloadEventType::kRequeue:
      return 3;
  }
  return 4;
}

class VectorResponseSink final : public ResponseSink {
 public:
  VectorResponseSink(std::vector<ResponseEvent>& responses, size_t capacity, ReplayWriter& replay,
                     const ManualClock& clock) noexcept
      : responses_(responses), capacity_(capacity), replay_(replay), clock_(clock) {}

  absl::StatusOr<ResponseReservation> Prepare(ResponseEvent event) override {
    if (!failure_.ok()) return failure_;
    if (responses_.size() >= capacity_) {
      return SimulationError(absl::StatusCode::kResourceExhausted, "response capacity exhausted",
                             ErrorReason::kQueueFull);
    }
    return ResponseReservation(event);
  }

  void Commit(ResponseReservation reservation) noexcept override {
    ResponseEvent event = reservation.Take();
    responses_.push_back(event);
    failure_ = replay_.WriteResponse(event, clock_.Now());
  }

  [[nodiscard]] const absl::Status& failure() const noexcept { return failure_; }

 private:
  std::vector<ResponseEvent>& responses_;
  size_t capacity_;
  ReplayWriter& replay_;
  const ManualClock& clock_;
  absl::Status failure_;
};

struct RequestOptions {
  TokenCount reservation_output_tokens{0};
  FinishReason success_reason = FinishReason::kLength;
};

}  // namespace

struct EngineSimulator::Impl {
  Impl(config::EngineConfig config, config::ModelCapabilities model, ReplaySink& replay,
       std::unique_ptr<scheduler::StepPlanPool> plan_pool, std::unique_ptr<FakeExecutor> executor)
      : config_(config),
        model_(model),
        replay_sink_(replay),
        clock_(MonotonicTime{}),
        resources_(SequenceCount(static_cast<uint32_t>(config_.MaxActiveSequences())),
                   KvTokenCount(config_.SimulatedKvTokenCapacity())),
        plan_pool_(std::move(plan_pool)),
        executor_(std::move(executor)) {}

  [[nodiscard]] absl::Status InitializeRun() {
    replay_writer_ = std::make_unique<ReplayWriter>(replay_sink_, config_.CanonicalJson(), model_,
                                                    workload_.size());
    response_sink_ = std::make_unique<VectorResponseSink>(responses_, response_capacity_,
                                                          *replay_writer_, clock_);
    coordinator_ = std::make_unique<LifecycleCoordinator>(
        registry_, resources_, *response_sink_, *replay_writer_,
        static_cast<size_t>(config_.MaxQueuedRequests()));
    invariant_checker_ = std::make_unique<InvariantChecker>(
        registry_, resources_, *plan_pool_, *executor_, responses_, model_.max_context_tokens);
    absl::Status header = replay_writer_->WriteHeader();
    if (!header.ok()) return header;
    for (const WorkloadEvent& event : workload_) {
      absl::Status written = replay_writer_->WriteWorkload(event);
      if (!written.ok()) return written;
    }
    return absl::OkStatus();
  }

  [[nodiscard]] bool HasUnfinishedRequests() const {
    std::vector<const RequestContext*> order;
    registry_.AppendArrivalOrder(order);
    return std::any_of(order.begin(), order.end(),
                       [](const RequestContext* request) { return !IsTerminal(request->state); });
  }

  [[nodiscard]] std::optional<MonotonicTime> NextDeadline() const {
    std::optional<MonotonicTime> next;
    std::vector<const RequestContext*> order;
    registry_.AppendArrivalOrder(order);
    for (const RequestContext* request : order) {
      const std::optional<MonotonicTime> deadline = request->request->deadline;
      if (IsTerminal(request->state) || !deadline.has_value()) continue;
      if (!next.has_value() || deadline.value() < next.value()) {
        next = deadline;
      }
    }
    return next;
  }

  [[nodiscard]] std::optional<MonotonicTime> NextTimestamp() const {
    std::optional<MonotonicTime> next;
    if (workload_index_ < workload_.size()) next = workload_[workload_index_].at;
    const std::optional<MonotonicTime> deadline = NextDeadline();
    if (deadline.has_value() && (!next.has_value() || *deadline < *next)) next = deadline;
    const std::optional<MonotonicTime> completion = executor_->NextCompletionTime();
    if (completion.has_value() && (!next.has_value() || *completion < *next)) next = completion;
    return next;
  }

  [[nodiscard]] absl::Status BumpEvent() {
    if (events_processed_ >= config_.MaxSimulationEvents()) {
      return SimulationError(absl::StatusCode::kResourceExhausted, "max_simulation_events exceeded",
                             ErrorReason::kEventLimit);
    }
    ++events_processed_;
    return absl::OkStatus();
  }

  [[nodiscard]] absl::Status ProcessWorkloadEvent(const WorkloadEvent& event) {
    absl::Status bumped = BumpEvent();
    if (!bumped.ok()) return bumped;
    switch (event.type) {
      case WorkloadEventType::kSubmit: {
        if (shutdown_) {
          return SimulationError(absl::StatusCode::kFailedPrecondition, "submission after shutdown",
                                 ErrorReason::kInvalidWorkload);
        }
        const SubmitWorkloadEvent& submitted = std::get<SubmitWorkloadEvent>(event.payload);
        const auto* tokens = std::get_if<std::vector<TokenId>>(&submitted.request.input);
        if (tokens == nullptr || tokens->size() > config_.MaxPromptTokens() ||
            submitted.request.generation.max_output_tokens.value() > config_.MaxOutputTokens() ||
            tokens->size() > config_.MaxScheduledTokensPerStep()) {
          return SimulationError(absl::StatusCode::kInvalidArgument,
                                 "request exceeds configured token limits",
                                 ErrorReason::kInvalidRequest);
        }
        const ModelRecord model_record{ModelId(0), model_.max_context_tokens, model_.accepts_text};
        absl::StatusOr<GenerateRequest> validated =
            ValidateGenerateRequest(submitted.request, model_record, clock_.Now());
        if (!validated.ok()) return validated.status();
        const TokenCount reservation_output = validated->generation.max_output_tokens;
        FinishReason success_reason = FinishReason::kLength;
        if (submitted.finish_after_tokens.has_value()) {
          validated->generation.max_output_tokens = *submitted.finish_after_tokens;
          success_reason = FinishReason::kSimulatedEos;
        }
        const RequestId id = validated->id;
        absl::Status admitted = coordinator_->Admit(std::move(*validated), event.at);
        if (!admitted.ok()) return admitted;
        RequestContext* context = registry_.Find(id);
        context->reservation_output_tokens = reservation_output;
        request_options_.emplace(id, RequestOptions{reservation_output, success_reason});
        ++requests_submitted_;
        return absl::OkStatus();
      }
      case WorkloadEventType::kCancel: {
        const RequestId request = std::get<CancelWorkloadEvent>(event.payload).request;
        return coordinator_->Cancel(
            request, FinishReason::kCancelled,
            WithErrorReason(absl::CancelledError("explicit workload cancellation"),
                            ErrorReason::kExplicitCancellation),
            ErrorReason::kExplicitCancellation, clock_.Now());
      }
      case WorkloadEventType::kInjectExecutorFailure:
        return executor_->AddFailureRule(std::get<InjectExecutorFailureEvent>(event.payload).rule);
      case WorkloadEventType::kPreempt:
        return coordinator_->Preempt(std::get<PreemptWorkloadEvent>(event.payload).request,
                                     clock_.Now());
      case WorkloadEventType::kRequeue:
        return coordinator_->Requeue(std::get<RequeueWorkloadEvent>(event.payload).request,
                                     clock_.Now());
      case WorkloadEventType::kShutdown: {
        shutdown_ = true;
        const ShutdownMode mode = std::get<ShutdownWorkloadEvent>(event.payload).mode;
        if (mode == ShutdownMode::kDrain) return absl::OkStatus();
        std::vector<const RequestContext*> order;
        registry_.AppendArrivalOrder(order);
        for (const RequestContext* request : order) {
          if (IsTerminal(request->state)) continue;
          absl::Status cancelled = coordinator_->Cancel(
              request->request->id, FinishReason::kShutdown,
              WithErrorReason(absl::CancelledError("simulator shutdown"), ErrorReason::kShutdown),
              ErrorReason::kShutdown, clock_.Now());
          if (!cancelled.ok()) return cancelled;
        }
        return absl::OkStatus();
      }
    }
    return SimulationError(absl::StatusCode::kInternal, "unknown workload event",
                           ErrorReason::kInvariantViolation);
  }

  [[nodiscard]] absl::Status ProcessDeadlines() {
    std::vector<const RequestContext*> order;
    registry_.AppendArrivalOrder(order);
    for (const RequestContext* request : order) {
      const std::optional<MonotonicTime> deadline = request->request->deadline;
      if (IsTerminal(request->state) || !deadline.has_value() || clock_.Now() < deadline.value()) {
        continue;
      }
      absl::Status bumped = BumpEvent();
      if (!bumped.ok()) return bumped;
      absl::Status expired = coordinator_->Cancel(
          request->request->id, FinishReason::kDeadline,
          WithErrorReason(absl::DeadlineExceededError("request deadline expired"),
                          ErrorReason::kDeadlineExpired),
          ErrorReason::kDeadlineExpired, clock_.Now());
      if (!expired.ok()) return expired;
    }
    return absl::OkStatus();
  }

  [[nodiscard]] absl::Status ProcessCompletions() {
    while (true) {
      absl::StatusOr<size_t> count = executor_->CompleteReady(clock_.Now(), completion_scratch_);
      if (!count.ok()) return count.status();
      if (*count == 0) return absl::OkStatus();
      const ExecutionTicketId ticket = completion_scratch_[0].ticket;
      for (size_t index = 0; index < *count; ++index) {
        absl::Status bumped = BumpEvent();
        if (!bumped.ok()) return bumped;
        const ExecutionCompletion& completion = completion_scratch_[index];
        absl::Status written = replay_writer_->WriteCompletion(completion, clock_.Now());
        if (!written.ok()) return written;
        absl::Status applied = coordinator_->Complete(completion, clock_.Now());
        if (!applied.ok()) return applied;
        RequestContext* request = registry_.Find(completion.request);
        if (request != nullptr && request->state == RequestState::kFinishing) {
          const auto option = request_options_.find(completion.request);
          const FinishReason finish = option == request_options_.end()
                                          ? FinishReason::kLength
                                          : option->second.success_reason;
          absl::Status terminal =
              coordinator_->EmitTerminal(completion.request, finish, clock_.Now());
          if (!terminal.ok()) return terminal;
        }
      }
      absl::Status acknowledged = executor_->Acknowledge(ticket);
      if (!acknowledged.ok()) return acknowledged;
    }
  }

  void BuildViews(RequestState first, RequestState second,
                  std::vector<scheduler::RequestSchedulingView>& output) {
    all_views_.clear();
    registry_.AppendSchedulingViews(all_views_);
    output.clear();
    for (const scheduler::RequestSchedulingView& view : all_views_) {
      if (view.state == first || view.state == second) output.push_back(view);
    }
  }

  [[nodiscard]] absl::Status PlanOneStep() {
    BuildViews(RequestState::kQueued, RequestState::kQueued, admission_views_);
    bool capacity_blocked = false;
    if (!admission_views_.empty()) {
      scheduler::SchedulingSnapshot admission_snapshot{
          .now = clock_.Now(),
          .proposed_step = StepId(next_step_id_),
          .requests = admission_views_,
          .limits =
              scheduler::SchedulingLimits{
                  SequenceCount(static_cast<uint32_t>(config_.MaxSequencesPerStep())),
                  TokenCount(static_cast<uint32_t>(config_.MaxScheduledTokensPerStep()))},
          .resources = resources_.Snapshot(),
      };
      absl::StatusOr<size_t> selected = policy_.Select(admission_snapshot, selection_scratch_);
      if (!selected.ok()) return selected.status();
      for (size_t position = 0; position < *selected; ++position) {
        const scheduler::RequestSchedulingView& request =
            admission_views_[selection_scratch_[position]];
        const scheduler::AdmissionDecision decision =
            admission_.Evaluate(request, resources_.Snapshot());
        if (decision.kind == scheduler::AdmissionKind::kNeverFits) {
          absl::Status failed = coordinator_->Fail(request.request, decision.rejection,
                                                   ErrorReason::kImpossibleContext, clock_.Now());
          if (!failed.ok()) return failed;
          continue;
        }
        if (decision.kind == scheduler::AdmissionKind::kTemporarilyBlocked) {
          capacity_blocked = true;
          break;
        }
        absl::Status reserved = coordinator_->Reserve(request.request, decision.cost, clock_.Now());
        if (!reserved.ok()) return reserved;
      }
    }

    BuildViews(RequestState::kPrefillReady, RequestState::kDecodeReady, ready_views_);
    if (ready_views_.empty()) {
      if (HasUnfinishedRequests()) {
        const IdleReason reason = capacity_blocked ? IdleReason::kCapacityBlocked
                                  : executor_->live_tickets() > 0
                                      ? IdleReason::kWaitingForCompletion
                                      : IdleReason::kWaitingForEvent;
        return replay_writer_->WriteIdle(reason, clock_.Now(), NextTimestamp());
      }
      return absl::OkStatus();
    }
    if (next_step_id_ == std::numeric_limits<uint64_t>::max()) {
      return SimulationError(absl::StatusCode::kOutOfRange, "step id exhausted",
                             ErrorReason::kInvariantViolation);
    }
    scheduler::SchedulingSnapshot snapshot{
        .now = clock_.Now(),
        .proposed_step = StepId(next_step_id_),
        .requests = ready_views_,
        .limits =
            scheduler::SchedulingLimits{
                SequenceCount(static_cast<uint32_t>(config_.MaxSequencesPerStep())),
                TokenCount(static_cast<uint32_t>(config_.MaxScheduledTokensPerStep()))},
        .resources = resources_.Snapshot(),
    };
    absl::StatusOr<size_t> selected = policy_.Select(snapshot, selection_scratch_);
    if (!selected.ok()) return selected.status();
    absl::StatusOr<scheduler::StepPlanLease> lease =
        plan_pool_->Acquire(snapshot.proposed_step, ModelId(0), snapshot.now);
    if (!lease.ok()) {
      if (lease.status().code() == absl::StatusCode::kResourceExhausted) {
        return replay_writer_->WriteIdle(IdleReason::kNoPlanBuffer, clock_.Now(),
                                         executor_->NextCompletionTime());
      }
      return lease.status();
    }
    absl::Status built = planner_.Build(
        snapshot, std::span<const size_t>(selection_scratch_.data(), *selected), *lease);
    if (!built.ok()) return built;
    if (lease->plan().sequences.empty()) {
      return replay_writer_->WriteIdle(IdleReason::kCapacityBlocked, clock_.Now(), NextTimestamp());
    }
    absl::Status valid = validator_.Validate(lease->plan(), snapshot);
    if (!valid.ok()) return valid;
    absl::Status recorded = replay_writer_->WritePlan(lease->plan());
    if (!recorded.ok()) return recorded;
    const scheduler::StepPlan& accepted_plan = lease->plan();
    absl::StatusOr<ExecutionTicket> ticket = executor_->Submit(std::move(*lease));
    if (!ticket.ok()) return ticket.status();
    absl::Status submitted = coordinator_->MarkSubmitted(accepted_plan, *ticket, clock_.Now());
    if (!submitted.ok()) return submitted;
    ++steps_submitted_;
    ++next_step_id_;
    return BumpEvent();
  }

  [[nodiscard]] absl::Status StepOneTimestamp() {
    if (!running_ || finished_) {
      return SimulationError(absl::StatusCode::kFailedPrecondition,
                             "StepOneTimestamp requires an active run",
                             ErrorReason::kInvalidTransition);
    }
    const std::optional<MonotonicTime> next = NextTimestamp();
    if (!next.has_value()) {
      if (HasUnfinishedRequests()) {
        return SimulationError(absl::StatusCode::kInternal,
                               "unfinished requests have no future event", ErrorReason::kDeadlock);
      }
      finished_ = true;
      return absl::OkStatus();
    }
    if (*next < clock_.Now()) {
      return SimulationError(absl::StatusCode::kInternal, "time moved backwards",
                             ErrorReason::kInvariantViolation);
    }
    absl::StatusOr<MonotonicTime> advanced = clock_.Advance(*next - clock_.Now());
    if (!advanced.ok()) return advanced.status();

    while (workload_index_ < workload_.size() && workload_[workload_index_].at == clock_.Now() &&
           EventClass(workload_[workload_index_].type) == 1) {
      absl::Status status = ProcessWorkloadEvent(workload_[workload_index_++]);
      if (!status.ok()) return status;
    }
    absl::Status deadlines = ProcessDeadlines();
    if (!deadlines.ok()) return deadlines;
    while (workload_index_ < workload_.size() && workload_[workload_index_].at == clock_.Now() &&
           EventClass(workload_[workload_index_].type) == 2) {
      absl::Status status = ProcessWorkloadEvent(workload_[workload_index_++]);
      if (!status.ok()) return status;
    }
    absl::Status completions = ProcessCompletions();
    if (!completions.ok()) return completions;
    while (workload_index_ < workload_.size() && workload_[workload_index_].at == clock_.Now() &&
           EventClass(workload_[workload_index_].type) == 3) {
      absl::Status status = ProcessWorkloadEvent(workload_[workload_index_++]);
      if (!status.ok()) return status;
    }
    absl::Status planned = PlanOneStep();
    if (!planned.ok()) return planned;
    if (!response_sink_->failure().ok()) return response_sink_->failure();
    return invariant_checker_->Validate();
  }

  void Cleanup() noexcept {
    std::vector<const RequestContext*> order;
    registry_.AppendArrivalOrder(order);
    for (const RequestContext* request : order) {
      if (IsTerminal(request->state)) continue;
      if (request->state == RequestState::kFinishing) {
        static_cast<void>(coordinator_->EmitTerminal(request->request->id, FinishReason::kShutdown,
                                                     clock_.Now()));
      } else {
        static_cast<void>(coordinator_->Cancel(
            request->request->id, FinishReason::kShutdown,
            WithErrorReason(absl::CancelledError("simulator cleanup"), ErrorReason::kShutdown),
            ErrorReason::kShutdown, clock_.Now()));
      }
    }
    while (const std::optional<MonotonicTime> next = executor_->NextCompletionTime()) {
      if (*next > clock_.Now()) {
        static_cast<void>(clock_.Advance(*next - clock_.Now()));
      }
      absl::StatusOr<size_t> count = executor_->CompleteReady(clock_.Now(), completion_scratch_);
      if (!count.ok() || *count == 0) break;
      const ExecutionTicketId ticket = completion_scratch_[0].ticket;
      for (size_t index = 0; index < *count; ++index) {
        static_cast<void>(coordinator_->Complete(completion_scratch_[index], clock_.Now()));
      }
      static_cast<void>(executor_->Acknowledge(ticket));
    }
  }

  [[nodiscard]] SimulationSummary Summary(bool invariants_ok) const {
    SimulationSummary summary;
    summary.events = events_processed_;
    summary.steps = steps_submitted_;
    summary.requests = requests_submitted_;
    std::vector<const RequestContext*> order;
    registry_.AppendArrivalOrder(order);
    for (const RequestContext* request : order) {
      switch (request->state) {
        case RequestState::kFinished:
          ++summary.finished;
          break;
        case RequestState::kCancelled:
          ++summary.cancelled;
          break;
        case RequestState::kFailed:
          ++summary.failed;
          break;
        default:
          break;
      }
    }
    const scheduler::ResourceSnapshot resources = resources_.Snapshot();
    summary.used_sequences = resources.sequences_used.value();
    summary.used_kv_tokens = resources.kv_tokens_used.value();
    summary.live_tickets = executor_->live_tickets();
    summary.leased_plan_slots = plan_pool_->leased();
    summary.invariants_ok = invariants_ok;
    return summary;
  }

  config::EngineConfig config_;
  config::ModelCapabilities model_;
  ReplaySink& replay_sink_;
  ManualClock clock_;
  RequestRegistry registry_;
  scheduler::ResourceAccountant resources_;
  std::unique_ptr<scheduler::StepPlanPool> plan_pool_;
  scheduler::FcfsPolicy policy_;
  scheduler::AdmissionController admission_;
  scheduler::BatchPlanner planner_;
  scheduler::StepPlanValidator validator_;
  std::vector<scheduler::RequestSchedulingView> all_views_;
  std::vector<scheduler::RequestSchedulingView> admission_views_;
  std::vector<scheduler::RequestSchedulingView> ready_views_;
  std::vector<size_t> selection_scratch_;
  std::vector<ExecutionCompletion> completion_scratch_;
  std::unique_ptr<FakeExecutor> executor_;
  std::vector<WorkloadEvent> workload_;
  size_t workload_index_ = 0;
  std::vector<ResponseEvent> responses_;
  std::map<RequestId, RequestOptions> request_options_;
  std::unique_ptr<ReplayWriter> replay_writer_;
  std::unique_ptr<VectorResponseSink> response_sink_;
  std::unique_ptr<LifecycleCoordinator> coordinator_;
  std::unique_ptr<InvariantChecker> invariant_checker_;
  size_t response_capacity_ = 0;
  uint64_t next_step_id_ = 1;
  uint64_t events_processed_ = 0;
  uint64_t steps_submitted_ = 0;
  uint64_t requests_submitted_ = 0;
  bool loaded_ = false;
  bool running_ = false;
  bool finished_ = false;
  bool shutdown_ = false;
};

absl::StatusOr<std::unique_ptr<EngineSimulator>> EngineSimulator::Create(
    config::EngineConfig config, config::ModelCapabilities model, ReplaySink& replay) {
  if (model.max_context_tokens.value() == 0 ||
      config.MaxModelTokens() > model.max_context_tokens.value()) {
    return SimulationError(absl::StatusCode::kInvalidArgument,
                           "model capability is incompatible with config",
                           ErrorReason::kInvalidConfig);
  }
  auto pool = scheduler::StepPlanPool::Create(
      static_cast<uint32_t>(config.PlanBufferSlots()),
      SequenceCount(static_cast<uint32_t>(config.MaxSequencesPerStep())));
  if (!pool.ok()) return pool.status();
  auto executor = FakeExecutor::Create(
      LatencyModel(LatencyModelParameters{
          .base = Nanoseconds(static_cast<int64_t>(config.FakeBaseLatencyNs())),
          .prefill_per_token =
              Nanoseconds(static_cast<int64_t>(config.FakePrefillLatencyPerTokenNs())),
          .decode_per_sequence =
              Nanoseconds(static_cast<int64_t>(config.FakeDecodeLatencyPerSequenceNs())),
      }),
      static_cast<size_t>(config.PlanBufferSlots()),
      static_cast<size_t>(config.MaxQueuedRequests()));
  if (!executor.ok()) return executor.status();
  try {
    auto impl = std::make_unique<Impl>(config, model, replay, std::move(*pool),
                                       std::move(*executor));
    const size_t max_requests = static_cast<size_t>(impl->config_.MaxQueuedRequests());
    const size_t max_sequences = static_cast<size_t>(impl->config_.MaxSequencesPerStep());
    impl->all_views_.reserve(max_requests);
    impl->admission_views_.reserve(max_requests);
    impl->ready_views_.reserve(max_requests);
    impl->selection_scratch_.resize(max_requests);
    impl->completion_scratch_.resize(max_sequences);
    impl->workload_.reserve(max_requests);
    impl->request_options_.clear();
    return std::unique_ptr<EngineSimulator>(new EngineSimulator(std::move(impl)));
  } catch (const std::bad_alloc&) {
    return SimulationError(absl::StatusCode::kResourceExhausted,
                           "bounded simulator allocation failed", ErrorReason::kCapacityExhausted);
  } catch (...) {
    return SimulationError(absl::StatusCode::kInternal,
                           "unexpected simulator construction exception",
                           ErrorReason::kInvariantViolation);
  }
}

EngineSimulator::EngineSimulator(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
EngineSimulator::~EngineSimulator() = default;

absl::Status EngineSimulator::LoadWorkload(std::span<const WorkloadEvent> events) {
  if (impl_->loaded_ || impl_->running_) {
    return SimulationError(absl::StatusCode::kFailedPrecondition,
                           "workload may be loaded exactly once before Run",
                           ErrorReason::kInvalidTransition);
  }
  uint64_t maximum_responses = 0;
  uint64_t submit_count = 0;
  MonotonicTime previous{};
  bool have_previous = false;
  for (const WorkloadEvent& event : events) {
    if (have_previous && event.at < previous) {
      return SimulationError(absl::StatusCode::kInvalidArgument,
                             "workload is not in total timestamp order",
                             ErrorReason::kInvalidWorkload);
    }
    previous = event.at;
    have_previous = true;
    if (event.type == WorkloadEventType::kSubmit) {
      const SubmitWorkloadEvent& submit = std::get<SubmitWorkloadEvent>(event.payload);
      absl::StatusOr<uint64_t> next =
          CheckedAdd(maximum_responses,
                     static_cast<uint64_t>(submit.request.generation.max_output_tokens.value()),
                     "simulator.workload.response_capacity");
      if (!next.ok()) return next.status();
      maximum_responses = *next;
      ++submit_count;
    }
  }
  absl::StatusOr<uint64_t> maximum_events =
      CheckedAdd(maximum_responses, submit_count, "simulator.workload.event_capacity");
  if (!maximum_events.ok()) return maximum_events.status();
  if (*maximum_events > impl_->config_.MaxSimulationEvents()) {
    return SimulationError(absl::StatusCode::kResourceExhausted,
                           "workload cannot fit max_simulation_events", ErrorReason::kEventLimit);
  }
  try {
    impl_->workload_.assign(events.begin(), events.end());
    impl_->response_capacity_ = static_cast<size_t>(*maximum_events);
    impl_->responses_.reserve(impl_->response_capacity_);
  } catch (const std::bad_alloc&) {
    impl_->workload_.clear();
    return SimulationError(absl::StatusCode::kResourceExhausted,
                           "workload bounded storage allocation failed",
                           ErrorReason::kCapacityExhausted);
  }
  impl_->loaded_ = true;
  return absl::OkStatus();
}

absl::StatusOr<SimulationResult> EngineSimulator::Run() {
  if (!impl_->loaded_ || impl_->running_) {
    return SimulationError(absl::StatusCode::kFailedPrecondition,
                           "Run requires one loaded, unused workload",
                           ErrorReason::kInvalidTransition);
  }
  impl_->running_ = true;
  absl::Status status = impl_->InitializeRun();
  while (status.ok() && !impl_->finished_) {
    status = impl_->StepOneTimestamp();
  }
  if (!status.ok()) impl_->Cleanup();
  if (status.ok() && impl_->executor_->unconsumed_failure_rules() != 0) {
    status =
        SimulationError(absl::StatusCode::kFailedPrecondition,
                        "one-shot failure rule was not consumed", ErrorReason::kInvalidWorkload);
    impl_->Cleanup();
  }
  absl::Status invariant_status = impl_->invariant_checker_->ValidateFinal();
  const bool invariants_ok = invariant_status.ok();
  if (status.ok() && !invariant_status.ok()) status = invariant_status;
  SimulationSummary summary = impl_->Summary(invariants_ok);
  if (status.ok() && summary.failed != 0) {
    status = SimulationError(absl::StatusCode::kInternal, "one or more requests failed execution",
                             ErrorReason::kExecutorFailure);
  }
  ErrorReason reason = ErrorReason::kNone;
  if (!status.ok()) {
    reason = GetErrorReason(status).value_or(ErrorReason::kInvariantViolation);
  }
  ReplayFooter footer{
      .final_status = status.code(),
      .error_reason = reason,
      .events = summary.events,
      .steps = summary.steps,
      .requests = summary.requests,
      .finished = summary.finished,
      .cancelled = summary.cancelled,
      .failed = summary.failed,
      .used_sequences = summary.used_sequences,
      .used_kv_tokens = summary.used_kv_tokens,
      .live_tickets = summary.live_tickets,
      .leased_plan_slots = summary.leased_plan_slots,
      .invariants_ok = summary.invariants_ok,
  };
  absl::Status footer_status = impl_->replay_writer_->WriteFooter(footer);
  if (status.ok() && !footer_status.ok()) status = footer_status;
  impl_->running_ = false;
  return SimulationResult{std::move(status), summary};
}

absl::Status EngineSimulator::StepOneTimestamp() { return impl_->StepOneTimestamp(); }

std::span<const ResponseEvent> EngineSimulator::responses() const noexcept {
  return impl_->responses_;
}

}  // namespace inferx::simulator
