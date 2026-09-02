#include "inferx/runtime/single_request_runner.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/api/generate_request.h"
#include "inferx/api/response_event.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/engine/event_sink.h"
#include "inferx/engine/execution_completion.h"
#include "inferx/engine/lifecycle_coordinator.h"
#include "inferx/engine/request_context.h"
#include "inferx/engine/request_registry.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/runtime/execution_backend.h"
#include "inferx/runtime/model_handle.h"
#include "inferx/scheduler/batch_planner.h"
#include "inferx/scheduler/fcfs_policy.h"
#include "inferx/scheduler/resource_accountant.h"
#include "inferx/scheduler/scheduling_snapshot.h"
#include "inferx/scheduler/step_plan.h"
#include "inferx/scheduler/step_plan_validator.h"

namespace inferx::runtime {
namespace {

constexpr uint32_t kPlanBufferSlots = 2;

absl::Status RunnerError(absl::StatusCode code, std::string_view detail) {
  return absl::Status(code, absl::StrCat("single_request_runner: ", detail));
}

}  // namespace

SingleRequestRunner::SingleRequestRunner(ExecutionBackend& backend, ModelHandle model,
                                         uint64_t max_prefill_tokens,
                                         std::unique_ptr<scheduler::ResourceAccountant> resources,
                                         std::unique_ptr<RequestRegistry> registry,
                                         std::unique_ptr<LifecycleCoordinator> coordinator,
                                         std::shared_ptr<scheduler::StepPlanPool> plan_pool)
    : backend_(backend),
      model_(model),
      max_prefill_tokens_(max_prefill_tokens),
      resources_(std::move(resources)),
      registry_(std::move(registry)),
      coordinator_(std::move(coordinator)),
      plan_pool_(std::move(plan_pool)) {}

SingleRequestRunner::~SingleRequestRunner() = default;

absl::StatusOr<std::unique_ptr<SingleRequestRunner>> SingleRequestRunner::Create(
    ExecutionBackend& backend, ModelHandle model, uint64_t context_capacity,
    uint64_t max_prefill_tokens, ResponseSink& sink) {
  if (context_capacity == 0 || max_prefill_tokens == 0 || max_prefill_tokens > context_capacity) {
    return RunnerError(absl::StatusCode::kInvalidArgument, "invalid execution envelope");
  }
  auto resources = std::make_unique<scheduler::ResourceAccountant>(SequenceCount(1),
                                                                   KvTokenCount(context_capacity));
  auto registry = std::make_unique<RequestRegistry>();
  static NullLifecycleObserver null_observer;
  auto coordinator = std::make_unique<LifecycleCoordinator>(*registry, *resources, sink,
                                                            null_observer, /*max_live_requests=*/1);
  auto plan_pool = scheduler::StepPlanPool::Create(kPlanBufferSlots, SequenceCount(1));
  if (!plan_pool.ok()) return plan_pool.status();
  return std::unique_ptr<SingleRequestRunner>(
      new SingleRequestRunner(backend, model, max_prefill_tokens, std::move(resources),
                              std::move(registry), std::move(coordinator), std::move(*plan_pool)));
}

absl::StatusOr<SingleRequestRunner::Result> SingleRequestRunner::Run(GenerateRequest request,
                                                                     MonotonicTime now) {
  const RequestId id = request.id;
  const size_t max_output = request.generation.max_output_tokens.value();

  absl::Status status = coordinator_->Admit(std::move(request), now);
  if (!status.ok()) return status;
  const RequestContext* context = registry_->Find(id);
  if (context == nullptr) return RunnerError(absl::StatusCode::kInternal, "admission lost request");
  const uint64_t prompt = context->prompt_tokens.size();

  // Conservative M5 reservation: one sequence and prompt plus maximum
  // output tokens for the full request lifetime.
  const uint64_t cost = prompt + max_output;
  if (cost > std::numeric_limits<uint32_t>::max()) {
    return RunnerError(absl::StatusCode::kInvalidArgument, "context cost exceeds token count");
  }
  status = coordinator_->Reserve(
      id, scheduler::ResourceCost{SequenceCount(1), TokenCount(static_cast<uint32_t>(cost))}, now);
  if (!status.ok()) return status;

  Result result;
  result.prompt_tokens = TokenCount(static_cast<uint32_t>(prompt));
  while (true) {
    context = registry_->Find(id);
    if (context == nullptr) return RunnerError(absl::StatusCode::kInternal, "request vanished");
    if (IsTerminal(context->state) || context->state == RequestState::kFinishing) {
      status = FinishIfTerminal(id, now, result);
      if (!status.ok()) return status;
      return result;
    }
    if (context->state != RequestState::kPrefillReady &&
        context->state != RequestState::kDecodeReady) {
      return RunnerError(absl::StatusCode::kInternal, "unexpected ready state");
    }
    status = PlanSubmitComplete(id, now);
    if (!status.ok()) return status;

    // A committed output that matched a stop id ends generation now, before
    // another decode consumes it (m5.md section 13.4). The token delta is
    // already committed; exactly one terminal response follows.
    context = registry_->Find(id);
    if (context == nullptr) return RunnerError(absl::StatusCode::kInternal, "request vanished");
    if (!context->output_tokens.empty() && (context->state == RequestState::kPrefillReady ||
                                            context->state == RequestState::kDecodeReady)) {
      const TokenId last = context->output_tokens.back();
      bool matched_stop = false;
      for (const int32_t stop : backend_.stop_token_ids(model_)) {
        if (stop == last.value()) matched_stop = true;
      }
      if (matched_stop) {
        status = coordinator_->Stop(id, now);
        if (!status.ok()) return status;
        status = coordinator_->EmitTerminal(id, FinishReason::kEos, now);
        if (!status.ok()) return status;
      }
    }
  }
}

absl::Status SingleRequestRunner::PlanSubmitComplete(RequestId request, MonotonicTime now) {
  std::vector<scheduler::RequestSchedulingView> views;
  registry_->AppendSchedulingViews(views);
  const scheduler::RequestSchedulingView* selected = nullptr;
  for (const auto& view : views) {
    if (view.request == request &&
        (view.state == RequestState::kPrefillReady || view.state == RequestState::kDecodeReady)) {
      selected = &view;
    }
  }
  if (selected == nullptr) {
    return RunnerError(absl::StatusCode::kInternal, "no ready scheduling view");
  }

  const scheduler::SchedulingLimits limits{SequenceCount(1),
                                           TokenCount(static_cast<uint32_t>(max_prefill_tokens_))};
  const std::array<scheduler::RequestSchedulingView, 1> ready{*selected};
  const scheduler::SchedulingSnapshot snapshot{
      now,
      StepId(next_step_.value()),
      std::span<const scheduler::RequestSchedulingView>(ready.data(), ready.size()),
      limits,
      resources_->Snapshot(),
  };

  auto lease = plan_pool_->Acquire(snapshot.proposed_step, ModelId(0), now);
  if (!lease.ok()) return lease.status();
  const size_t selection = 0;
  absl::Status status = planner_.Build(snapshot, std::span<const size_t>(&selection, 1), *lease);
  if (!status.ok()) return status;
  status = validator_.Validate(lease->plan(), snapshot);
  if (!status.ok()) return status;

  // The step's input-token values: the prompt for prefill, the previously
  // emitted token for decode. Copied into ticket-owned backend metadata.
  const RequestContext* context = registry_->Find(request);
  if (context == nullptr) return RunnerError(absl::StatusCode::kInternal, "request vanished");
  const scheduler::ScheduledSequence& item = lease->plan().sequences.front();
  std::vector<int32_t> step_tokens;
  if (item.kind == WorkKind::kPrefill) {
    step_tokens.reserve(context->prompt_tokens.size());
    for (const TokenId token : context->prompt_tokens) {
      step_tokens.push_back(token.value());
    }
  } else {
    if (context->output_tokens.empty()) {
      return RunnerError(absl::StatusCode::kInternal, "decode without a prior output token");
    }
    step_tokens.push_back(context->output_tokens.back().value());
  }

  const scheduler::StepPlan& accepted_plan = lease->plan();
  auto ticket = backend_.Submit(std::move(*lease), model_, step_tokens);
  if (!ticket.ok()) return ticket.status();
  status = coordinator_->MarkSubmitted(accepted_plan, *ticket, now);
  if (!status.ok()) return status;
  next_step_ = StepId(next_step_.value() + 1);

  ExecutionCompletion completion;
  while (true) {
    auto drained = backend_.CompleteReady(now, std::span<ExecutionCompletion>(&completion, 1));
    if (!drained.ok()) return drained.status();
    if (*drained == 0) continue;  // synchronous backend completes on submit
    break;
  }
  status = coordinator_->Complete(completion, now);
  if (!status.ok()) return status;
  return backend_.Acknowledge(completion.ticket);
}

absl::Status SingleRequestRunner::FinishIfTerminal(RequestId request, MonotonicTime now,
                                                   Result& result) {
  const RequestContext* context = registry_->Find(request);
  if (context == nullptr) return RunnerError(absl::StatusCode::kInternal, "request vanished");
  result.output_tokens = context->output_tokens;
  result.finish = context->terminal.has_value() ? context->terminal->reason : FinishReason::kLength;
  if (!context->terminal_emitted && context->state == RequestState::kFinishing) {
    absl::Status status = coordinator_->EmitTerminal(request, FinishReason::kLength, now);
    if (!status.ok()) return status;
  }
  context = registry_->Find(request);
  if (context == nullptr || !context->terminal_emitted) {
    return RunnerError(absl::StatusCode::kInternal, "terminal response missing");
  }
  if (context->terminal->status != absl::StatusCode::kOk) {
    result.status = absl::Status(static_cast<absl::StatusCode>(context->terminal->status),
                                 "generation finished with an error terminal response");
  }
  absl::Status erased = registry_->EraseTerminal(request);
  if (!erased.ok()) return erased;
  return absl::OkStatus();
}

}  // namespace inferx::runtime
