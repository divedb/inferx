#include <absl/status/statusor.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "inferx/api/generate_request.h"
#include "inferx/api/response_event.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"
#include "inferx/engine/execution_completion.h"
#include "inferx/engine/request_context.h"
#include "inferx/engine/request_controller.h"
#include "inferx/engine/request_event.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/scheduler/resource_accountant.h"
#include "inferx/scheduler/work_kind.h"
#include "inferx/simulator/fake_executor.h"

namespace {

struct Options {
  uint64_t seed = 0x4d31535452455353ULL;
  uint64_t operations = 50000;
  std::string failure_output;
};

enum class GeneratedAction : uint8_t {
  kSubmit,
  kCancel,
  kDeadline,
  kPreempt,
  kFailure,
  kShutdown,
  kInvalid,
  kCount,
};

constexpr size_t kGeneratedActionCount = static_cast<size_t>(GeneratedAction::kCount);

struct ReferenceRequest {
  inferx::RequestState state = inferx::RequestState::kReceived;
  uint32_t epoch = 0;
  uint32_t computed = 0;
  uint32_t scheduled = 0;
  uint32_t committed = 0;
  bool reservation = false;
  bool in_flight = false;
  bool pending_terminal = false;
  uint32_t terminal_count = 0;
};

struct GeneratedSlot {
  std::unique_ptr<inferx::RequestContext> context;
  ReferenceRequest reference;
  std::optional<inferx::ReservationId> reservation;
  inferx::scheduler::ResourceCost reservation_cost;
};

struct ContextDigest {
  inferx::RequestState state = inferx::RequestState::kReceived;
  inferx::RequestEpoch epoch{0};
  uint32_t computed = 0;
  uint32_t scheduled = 0;
  uint32_t committed = 0;
  size_t output_size = 0;
  std::optional<inferx::ReservationId> reservation;
  std::optional<inferx::StepId> step;
  std::optional<inferx::ExecutionTicketId> ticket;
  bool pending_terminal = false;
  bool terminal = false;
  bool terminal_emitted = false;

  friend bool operator==(const ContextDigest&, const ContextDigest&) = default;
};

ContextDigest Digest(const inferx::RequestContext& context) {
  return ContextDigest{
      .state = context.state,
      .epoch = context.epoch,
      .computed = context.num_computed_tokens,
      .scheduled = context.num_scheduled_tokens,
      .committed = context.num_committed_output_tokens,
      .output_size = context.output_tokens.size(),
      .reservation = context.reservation,
      .step = context.in_flight_step,
      .ticket = context.in_flight_ticket,
      .pending_terminal = context.pending_terminal_reason.has_value(),
      .terminal = context.terminal.has_value(),
      .terminal_emitted = context.terminal_emitted,
  };
}

class LifecycleGenerator {
 public:
  explicit LifecycleGenerator(uint64_t seed)
      : sequence_capacity_(static_cast<uint32_t>(seed % 8U) + 1U),
        kv_capacity_(static_cast<uint32_t>((seed >> 8U) % 224U) + 32U),
        accountant_(inferx::SequenceCount(sequence_capacity_), inferx::KvTokenCount(kv_capacity_)),
        slots_(static_cast<size_t>((seed >> 16U) % 17U) + 16U) {}

  [[nodiscard]] const char* Step(std::mt19937_64& random, uint64_t operation) {
    if (operation != 0 && operation % 4096U == 0) {
      ++coverage_[Index(GeneratedAction::kShutdown)];
      return Shutdown(operation);
    }

    GeneratedSlot& slot = slots_[static_cast<size_t>(random() % slots_.size())];
    if (slot.context != nullptr && operation % 19U == 0) {
      ++coverage_[Index(GeneratedAction::kInvalid)];
      return ApplyInvalid(slot);
    }
    if (slot.context == nullptr || inferx::IsTerminal(slot.context->state)) {
      ++coverage_[Index(GeneratedAction::kSubmit)];
      return SubmitNew(slot, random, operation);
    }

    const uint64_t choice = random() % 100U;
    switch (slot.reference.state) {
      case inferx::RequestState::kReceived:
        if (choice < 5U) return Cancel(slot, false, false, operation);
        if (choice < 10U) return Cancel(slot, true, false, operation);
        return InputReady(slot, operation);
      case inferx::RequestState::kQueued:
        if (choice < 5U) return Cancel(slot, false, false, operation);
        if (choice < 10U) return Cancel(slot, true, false, operation);
        return BeginReservation(slot, operation);
      case inferx::RequestState::kReserving:
        if (choice < 5U) return Cancel(slot, false, false, operation);
        if (choice < 10U) return Cancel(slot, true, false, operation);
        return ResolveReservation(slot, random, operation);
      case inferx::RequestState::kPrefillReady:
      case inferx::RequestState::kDecodeReady:
        if (choice < 10U) return Preempt(slot, operation);
        if (choice < 15U) return Cancel(slot, false, false, operation);
        if (choice < 20U) return Cancel(slot, true, false, operation);
        return SubmitWork(slot, operation);
      case inferx::RequestState::kPrefilling:
      case inferx::RequestState::kDecoding:
        if (choice < 8U) return Cancel(slot, false, false, operation);
        if (choice < 16U) return Cancel(slot, true, false, operation);
        if (choice < 25U) return ExecutionFailure(slot, operation);
        return Complete(slot, operation);
      case inferx::RequestState::kCancelling:
        return Drain(slot, operation);
      case inferx::RequestState::kPreempted:
        if (choice < 10U) return Cancel(slot, false, false, operation);
        if (choice < 20U) return Cancel(slot, true, false, operation);
        return Requeue(slot, operation);
      case inferx::RequestState::kFinishing:
        return Finish(slot, operation);
      case inferx::RequestState::kFinished:
      case inferx::RequestState::kCancelled:
      case inferx::RequestState::kFailed:
        return "terminal slot was not recycled";
      case inferx::RequestState::kTokenizing:
        return "generator entered unsupported tokenizing state";
    }
    return "generator reached unknown request state";
  }

  [[nodiscard]] const char* Finalize(uint64_t operation) {
    const char* error = Shutdown(operation);
    if (error != nullptr) return error;
    for (GeneratedSlot& slot : slots_) {
      if (slot.context == nullptr || inferx::IsTerminal(slot.context->state)) continue;
      if (slot.reference.state == inferx::RequestState::kCancelling) {
        error = Drain(slot, operation);
      } else if (slot.reference.state == inferx::RequestState::kFinishing) {
        error = Finish(slot, operation);
      } else {
        error = Cancel(slot, false, true, operation);
      }
      if (error != nullptr) return error;
      if (slot.reference.state == inferx::RequestState::kCancelling) {
        error = Drain(slot, operation);
        if (error != nullptr) return error;
      }
    }
    return CheckAll();
  }

  [[nodiscard]] const char* CheckCoverage(uint64_t operations) const {
    if (operations < 50000U) return nullptr;
    for (size_t index = 0; index < coverage_.size(); ++index) {
      if (coverage_[index] == 0) return "generated lifecycle category received zero coverage";
    }
    return nullptr;
  }

  void PrintCoverage() const {
    std::printf(
        " lifecycle_submit=%llu lifecycle_cancel=%llu lifecycle_deadline=%llu "
        "lifecycle_preempt=%llu lifecycle_failure=%llu lifecycle_shutdown=%llu "
        "lifecycle_invalid=%llu",
        Value(GeneratedAction::kSubmit), Value(GeneratedAction::kCancel),
        Value(GeneratedAction::kDeadline), Value(GeneratedAction::kPreempt),
        Value(GeneratedAction::kFailure), Value(GeneratedAction::kShutdown),
        Value(GeneratedAction::kInvalid));
  }

 private:
  static constexpr size_t Index(GeneratedAction action) { return static_cast<size_t>(action); }

  [[nodiscard]] unsigned long long Value(GeneratedAction action) const {
    return static_cast<unsigned long long>(coverage_[Index(action)]);
  }

  [[nodiscard]] const char* SubmitNew(GeneratedSlot& slot, std::mt19937_64& random,
                                      uint64_t operation) {
    const uint64_t id = next_request_id_++;
    const uint32_t prompt_count = static_cast<uint32_t>(random() % 12U) + 1U;
    const uint32_t output_count = static_cast<uint32_t>(random() % 8U) + 1U;
    std::vector<inferx::TokenId> tokens;
    tokens.reserve(prompt_count);
    for (uint32_t index = 0; index < prompt_count; ++index) {
      tokens.emplace_back(static_cast<int32_t>(index));
    }
    auto request = std::make_unique<inferx::GenerateRequest>(inferx::GenerateRequest{
        .id = inferx::RequestId(id),
        .model = inferx::ModelId(0),
        .input = std::move(tokens),
        .generation = inferx::GenerationLimits{inferx::TokenCount(output_count)},
        .priority = inferx::kDefaultPriority,
        .deadline = std::nullopt,
        .tenant = inferx::TenantScope(0),
    });
    auto context = std::make_unique<inferx::RequestContext>();
    context->request = std::move(request);
    context->sequence = inferx::SequenceId(id);
    context->arrival = inferx::ArrivalKey{
        inferx::MonotonicTime(inferx::Nanoseconds(static_cast<int64_t>(operation))), operation,
        inferx::RequestId(id)};
    context->reservation_output_tokens = inferx::TokenCount(output_count);
    context->output_tokens.reserve(output_count);
    slot = GeneratedSlot{
        .context = std::move(context),
        .reference = {},
        .reservation = std::nullopt,
        .reservation_cost = {},
    };
    return CheckAll();
  }

  [[nodiscard]] const char* InputReady(GeneratedSlot& slot, uint64_t operation) {
    const auto* tokens = std::get_if<std::vector<inferx::TokenId>>(&slot.context->request->input);
    if (tokens == nullptr) return "generated request lost token input";
    const char* error = Apply(slot,
                              inferx::RequestEvent{inferx::RequestEventKind::kInputReady,
                                                   inferx::InputReadyPayload{*tokens}},
                              operation);
    if (error != nullptr) return error;
    slot.reference.state = inferx::RequestState::kQueued;
    return CheckAll();
  }

  [[nodiscard]] const char* BeginReservation(GeneratedSlot& slot, uint64_t operation) {
    const char* error = Apply(slot,
                              inferx::RequestEvent{inferx::RequestEventKind::kBeginReservation,
                                                   inferx::BeginReservationPayload{}},
                              operation);
    if (error != nullptr) return error;
    slot.reference.state = inferx::RequestState::kReserving;
    return CheckAll();
  }

  [[nodiscard]] const char* ResolveReservation(GeneratedSlot& slot, std::mt19937_64& random,
                                               uint64_t operation) {
    const auto* tokens = std::get_if<std::vector<inferx::TokenId>>(&slot.context->request->input);
    if (tokens == nullptr) return "reservation request lost token input";
    uint32_t kv_tokens = static_cast<uint32_t>(tokens->size()) +
                         slot.context->request->generation.max_output_tokens.value();
    if (random() % 23U == 0) kv_tokens = kv_capacity_ + 1U;
    const inferx::scheduler::ResourceCost cost{inferx::SequenceCount(1),
                                               inferx::TokenCount(kv_tokens)};
    auto transaction = accountant_.BeginReservation(slot.context->request->id, cost);
    if (!transaction.ok()) {
      const auto before = accountant_.Snapshot();
      const char* error = Apply(slot,
                                inferx::RequestEvent{inferx::RequestEventKind::kReservationDeferred,
                                                     inferx::ReservationDeferredPayload{}},
                                operation);
      if (error != nullptr) return error;
      if (!(before == accountant_.Snapshot())) return "failed reservation mutated resources";
      slot.reference.state = inferx::RequestState::kQueued;
      return CheckAll();
    }
    const inferx::ReservationId reservation = transaction->Commit();
    const char* error =
        Apply(slot,
              inferx::RequestEvent{inferx::RequestEventKind::kReservationGranted,
                                   inferx::ReservationGrantedPayload{reservation, cost}},
              operation);
    if (error != nullptr) return error;
    slot.reservation = reservation;
    slot.reservation_cost = cost;
    slot.reference.reservation = true;
    slot.reference.state = slot.reference.computed == 0 ? inferx::RequestState::kPrefillReady
                                                        : inferx::RequestState::kDecodeReady;
    return CheckAll();
  }

  [[nodiscard]] const char* SubmitWork(GeneratedSlot& slot, uint64_t operation) {
    const bool prefill = slot.reference.state == inferx::RequestState::kPrefillReady;
    const uint32_t scheduled =
        prefill ? static_cast<uint32_t>(slot.context->prompt_tokens.size()) : 1U;
    const inferx::WorkKind work = prefill ? inferx::WorkKind::kPrefill : inferx::WorkKind::kDecode;
    const inferx::RequestEventKind kind = prefill ? inferx::RequestEventKind::kSubmitPrefill
                                                  : inferx::RequestEventKind::kSubmitDecode;
    const inferx::TokenRange range{inferx::TokenOffset(0), inferx::TokenOffset(scheduled)};
    const char* error = Apply(
        slot,
        inferx::RequestEvent{kind, inferx::SubmitPayload{inferx::StepId(next_step_id_),
                                                         inferx::ExecutionTicketId(next_ticket_id_),
                                                         inferx::RequestEpoch(slot.reference.epoch),
                                                         work, range, 1}},
        operation);
    ++next_step_id_;
    ++next_ticket_id_;
    if (error != nullptr) return error;
    slot.reference.state =
        prefill ? inferx::RequestState::kPrefilling : inferx::RequestState::kDecoding;
    slot.reference.scheduled = scheduled;
    slot.reference.in_flight = true;
    return CheckAll();
  }

  [[nodiscard]] inferx::ExecutionCompletion Completion(const GeneratedSlot& slot,
                                                       bool success) const {
    const inferx::RequestContext& context = *slot.context;
    return inferx::ExecutionCompletion{
        .ticket = context.in_flight_ticket.value_or(inferx::ExecutionTicketId(0)),
        .step = context.in_flight_step.value_or(inferx::StepId(0)),
        .item_ordinal = 0,
        .item_count = 1,
        .request = context.request->id,
        .sequence = context.sequence,
        .epoch = context.epoch,
        .kind = context.in_flight_work.value_or(inferx::WorkKind::kPrefill),
        .scheduled_tokens = context.in_flight_range.value_or(
            inferx::TokenRange{inferx::TokenOffset(0), inferx::TokenOffset(1)}),
        .status = success ? absl::OkStatus() : absl::InternalError("generated failure"),
        .error_reason =
            success ? inferx::ErrorReason::kNone : inferx::ErrorReason::kExecutorFailure,
        .output_token = success ? std::optional<inferx::TokenId>(inferx::simulator::FakeToken(
                                      context.request->id,
                                      inferx::TokenOffset(context.num_committed_output_tokens)))
                                : std::nullopt,
    };
  }

  [[nodiscard]] const char* Complete(GeneratedSlot& slot, uint64_t operation) {
    const bool prefill = slot.reference.state == inferx::RequestState::kPrefilling;
    const inferx::ExecutionCompletion completion = Completion(slot, true);
    const char* error =
        Apply(slot,
              prefill ? inferx::RequestEvent{inferx::RequestEventKind::kPrefillCompleted,
                                             inferx::PrefillCompletedPayload{completion}}
                      : inferx::RequestEvent{inferx::RequestEventKind::kDecodeCompleted,
                                             inferx::DecodeCompletedPayload{completion}},
              operation);
    if (error != nullptr) return error;
    slot.reference.computed += slot.reference.scheduled;
    slot.reference.scheduled = 0;
    ++slot.reference.committed;
    slot.reference.in_flight = false;
    slot.reference.state =
        slot.reference.committed < slot.context->request->generation.max_output_tokens.value()
            ? inferx::RequestState::kDecodeReady
            : inferx::RequestState::kFinishing;
    return CheckAll();
  }

  [[nodiscard]] const char* ExecutionFailure(GeneratedSlot& slot, uint64_t operation) {
    ++coverage_[Index(GeneratedAction::kFailure)];
    const inferx::ExecutionCompletion completion = Completion(slot, false);
    const char* error = Apply(slot,
                              inferx::RequestEvent{inferx::RequestEventKind::kExecutionFailed,
                                                   inferx::ExecutionFailedPayload{completion}},
                              operation);
    if (error != nullptr) return error;
    slot.reference.state = inferx::RequestState::kFailed;
    slot.reference.scheduled = 0;
    slot.reference.in_flight = false;
    slot.reference.terminal_count = 1;
    error = ReleaseReservation(slot);
    return error == nullptr ? CheckAll() : error;
  }

  [[nodiscard]] const char* Cancel(GeneratedSlot& slot, bool deadline, bool shutdown,
                                   uint64_t operation) {
    if (deadline) {
      ++coverage_[Index(GeneratedAction::kDeadline)];
    } else if (!shutdown) {
      ++coverage_[Index(GeneratedAction::kCancel)];
    }
    const inferx::FinishReason finish = shutdown   ? inferx::FinishReason::kShutdown
                                        : deadline ? inferx::FinishReason::kDeadline
                                                   : inferx::FinishReason::kCancelled;
    const inferx::ErrorReason reason = shutdown   ? inferx::ErrorReason::kShutdown
                                       : deadline ? inferx::ErrorReason::kDeadlineExpired
                                                  : inferx::ErrorReason::kExplicitCancellation;
    const absl::Status status =
        deadline ? WithReason(absl::DeadlineExceededError("generated deadline"), reason)
                 : WithReason(absl::CancelledError("generated cancellation"), reason);
    const inferx::RequestEventKind kind = deadline ? inferx::RequestEventKind::kDeadlineExpired
                                                   : inferx::RequestEventKind::kCancelRequested;
    const bool was_in_flight = slot.reference.in_flight;
    const char* error = Apply(
        slot, inferx::RequestEvent{kind, inferx::TerminalOutcomePayload{status, reason, finish}},
        operation);
    if (error != nullptr) return error;
    if (was_in_flight) {
      slot.reference.state = inferx::RequestState::kCancelling;
      slot.reference.pending_terminal = true;
    } else {
      slot.reference.state = inferx::RequestState::kCancelled;
      slot.reference.terminal_count = 1;
      if (slot.reference.reservation) {
        error = ReleaseReservation(slot);
        if (error != nullptr) return error;
      }
    }
    return CheckAll();
  }

  [[nodiscard]] const char* Drain(GeneratedSlot& slot, uint64_t operation) {
    const inferx::ExecutionCompletion completion = Completion(slot, false);
    const char* error = Apply(slot,
                              inferx::RequestEvent{inferx::RequestEventKind::kInFlightDrained,
                                                   inferx::InFlightDrainedPayload{completion}},
                              operation);
    if (error != nullptr) return error;
    slot.reference.state = inferx::RequestState::kCancelled;
    slot.reference.scheduled = 0;
    slot.reference.in_flight = false;
    slot.reference.pending_terminal = false;
    slot.reference.terminal_count = 1;
    error = ReleaseReservation(slot);
    return error == nullptr ? CheckAll() : error;
  }

  [[nodiscard]] const char* Preempt(GeneratedSlot& slot, uint64_t operation) {
    ++coverage_[Index(GeneratedAction::kPreempt)];
    const char* error = Apply(
        slot, inferx::RequestEvent{inferx::RequestEventKind::kPreempt, inferx::PreemptPayload{}},
        operation);
    if (error != nullptr) return error;
    slot.reference.state = inferx::RequestState::kPreempted;
    error = ReleaseReservation(slot);
    return error == nullptr ? CheckAll() : error;
  }

  [[nodiscard]] const char* Requeue(GeneratedSlot& slot, uint64_t operation) {
    const char* error = Apply(
        slot, inferx::RequestEvent{inferx::RequestEventKind::kRequeue, inferx::RequeuePayload{}},
        operation);
    if (error != nullptr) return error;
    slot.reference.state = inferx::RequestState::kQueued;
    ++slot.reference.epoch;
    return CheckAll();
  }

  [[nodiscard]] const char* Finish(GeneratedSlot& slot, uint64_t operation) {
    const char* error =
        Apply(slot,
              inferx::RequestEvent{inferx::RequestEventKind::kTerminalEmitted,
                                   inferx::TerminalEmittedPayload{inferx::FinishReason::kLength}},
              operation);
    if (error != nullptr) return error;
    slot.reference.state = inferx::RequestState::kFinished;
    slot.reference.terminal_count = 1;
    error = ReleaseReservation(slot);
    return error == nullptr ? CheckAll() : error;
  }

  [[nodiscard]] const char* Shutdown(uint64_t operation) {
    for (GeneratedSlot& slot : slots_) {
      if (slot.context == nullptr || inferx::IsTerminal(slot.context->state) ||
          slot.reference.state == inferx::RequestState::kCancelling) {
        continue;
      }
      const char* error = slot.reference.state == inferx::RequestState::kFinishing
                              ? Finish(slot, operation)
                              : Cancel(slot, false, true, operation);
      if (error != nullptr) return error;
    }
    return CheckAll();
  }

  [[nodiscard]] const char* Apply(GeneratedSlot& slot, const inferx::RequestEvent& event,
                                  uint64_t operation) {
    auto prepared = controller_.Prepare(*slot.context, event);
    if (!prepared.ok()) return "valid generated transition was rejected";
    controller_.StampTerminalTime(
        *prepared, inferx::MonotonicTime(inferx::Nanoseconds(static_cast<int64_t>(operation))));
    controller_.Commit(*slot.context, std::move(*prepared));
    return nullptr;
  }

  [[nodiscard]] const char* ApplyInvalid(GeneratedSlot& slot) {
    const ContextDigest before = Digest(*slot.context);
    const auto resources_before = accountant_.Snapshot();
    const inferx::RequestEvent event =
        slot.reference.state == inferx::RequestState::kFinishing
            ? inferx::RequestEvent{inferx::RequestEventKind::kRequeue, inferx::RequeuePayload{}}
            : inferx::RequestEvent{inferx::RequestEventKind::kTerminalEmitted,
                                   inferx::TerminalEmittedPayload{inferx::FinishReason::kLength}};
    if (controller_.Prepare(*slot.context, event).ok()) {
      return "intentionally invalid transition unexpectedly succeeded";
    }
    if (!(before == Digest(*slot.context)) || !(resources_before == accountant_.Snapshot())) {
      return "invalid transition changed state or resources";
    }
    return CheckAll();
  }

  [[nodiscard]] const char* ReleaseReservation(GeneratedSlot& slot) {
    if (!slot.reservation.has_value() || !accountant_.Release(slot.reservation.value()).ok()) {
      return "generated reservation release failed";
    }
    slot.reservation.reset();
    slot.reference.reservation = false;
    return nullptr;
  }

  [[nodiscard]] const char* CheckAll() const {
    uint64_t expected_sequences = 0;
    uint64_t expected_kv = 0;
    for (size_t slot_index = 0; slot_index < slots_.size(); ++slot_index) {
      const GeneratedSlot& slot = slots_[slot_index];
      if (slot.context == nullptr) continue;
      const inferx::RequestContext& context = *slot.context;
      const ReferenceRequest& reference = slot.reference;
      const bool all_in_flight =
          context.in_flight_step.has_value() && context.in_flight_ticket.has_value() &&
          context.submitted_ticket.has_value() && context.in_flight_work.has_value() &&
          context.in_flight_range.has_value();
      const bool no_in_flight =
          !context.in_flight_step.has_value() && !context.in_flight_ticket.has_value() &&
          !context.submitted_ticket.has_value() && !context.in_flight_work.has_value() &&
          !context.in_flight_range.has_value();
      if (context.state != reference.state || context.epoch.value() != reference.epoch ||
          context.num_computed_tokens != reference.computed ||
          context.num_scheduled_tokens != reference.scheduled ||
          context.num_committed_output_tokens != reference.committed ||
          context.output_tokens.size() != reference.committed ||
          context.reservation.has_value() != reference.reservation ||
          context.pending_terminal_reason.has_value() != reference.pending_terminal ||
          (reference.in_flight ? !all_in_flight : !no_in_flight) ||
          context.terminal_emitted != (reference.terminal_count == 1U) ||
          context.terminal.has_value() != (reference.terminal_count == 1U) ||
          reference.terminal_count > 1U) {
        std::fprintf(stderr,
                     "slot=%zu actual={state=%u epoch=%u computed=%u scheduled=%u committed=%u "
                     "output=%zu reservation=%u pending=%u inflight=%u terminal=%u emitted=%u} "
                     "reference={state=%u epoch=%u computed=%u scheduled=%u committed=%u "
                     "reservation=%u pending=%u inflight=%u terminal_count=%u}\n",
                     slot_index, static_cast<unsigned>(context.state), context.epoch.value(),
                     context.num_computed_tokens, context.num_scheduled_tokens,
                     context.num_committed_output_tokens, context.output_tokens.size(),
                     context.reservation.has_value() ? 1U : 0U,
                     context.pending_terminal_reason.has_value() ? 1U : 0U, all_in_flight ? 1U : 0U,
                     context.terminal.has_value() ? 1U : 0U, context.terminal_emitted ? 1U : 0U,
                     static_cast<unsigned>(reference.state), reference.epoch, reference.computed,
                     reference.scheduled, reference.committed, reference.reservation ? 1U : 0U,
                     reference.pending_terminal ? 1U : 0U, reference.in_flight ? 1U : 0U,
                     reference.terminal_count);
        return "generated lifecycle differs from reference model";
      }
      if (reference.reservation) {
        if (!slot.reservation.has_value() || context.reservation != slot.reservation) {
          return "generated reservation identity mismatch";
        }
        ++expected_sequences;
        expected_kv += slot.reservation_cost.kv_tokens.value();
      }
    }
    const inferx::scheduler::ResourceSnapshot snapshot = accountant_.Snapshot();
    if (!accountant_.Validate().ok() || snapshot.sequences_used.value() != expected_sequences ||
        snapshot.kv_tokens_used.value() != expected_kv ||
        snapshot.sequences_used.value() > sequence_capacity_ ||
        snapshot.kv_tokens_used.value() > kv_capacity_) {
      return "generated lifecycle resource model mismatch";
    }
    return nullptr;
  }

  static absl::Status WithReason(absl::Status status, inferx::ErrorReason reason) {
    return inferx::WithErrorReason(std::move(status), reason);
  }

  uint32_t sequence_capacity_;
  uint32_t kv_capacity_;
  inferx::scheduler::ResourceAccountant accountant_;
  std::vector<GeneratedSlot> slots_;
  inferx::RequestController controller_;
  std::array<uint64_t, kGeneratedActionCount> coverage_{};
  uint64_t next_request_id_ = 1;
  uint64_t next_step_id_ = 1;
  uint64_t next_ticket_id_ = 1;
};

std::optional<uint64_t> ParseInteger(std::string_view text) {
  try {
    size_t consumed = 0;
    const uint64_t value = std::stoull(std::string(text), &consumed, 0);
    if (consumed != text.size()) return std::nullopt;
    return value;
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<Options> ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument.starts_with("--seed=")) {
      const auto value = ParseInteger(argument.substr(7));
      if (!value) return std::nullopt;
      options.seed = *value;
    } else if (argument.starts_with("--operations=")) {
      const auto value = ParseInteger(argument.substr(13));
      if (!value || *value == 0) return std::nullopt;
      options.operations = *value;
    } else if (argument.starts_with("--failure_output=")) {
      options.failure_output = std::string(argument.substr(17));
    } else {
      return std::nullopt;
    }
  }
  return options;
}

int Fail(const Options& options, uint64_t operation, std::string_view message) {
  const uint64_t reproduce_operations =
      operation < options.operations ? operation + 1U : options.operations;
  std::fprintf(stderr, "seed=0x%llx operation=%llu failure=%.*s\n",
               static_cast<unsigned long long>(options.seed),
               static_cast<unsigned long long>(operation), static_cast<int>(message.size()),
               message.data());
  if (!options.failure_output.empty()) {
    std::error_code error;
    std::filesystem::create_directories(options.failure_output, error);
    std::ofstream output(std::filesystem::path(options.failure_output) / "failure.txt");
    output << "seed=0x" << std::hex << options.seed << std::dec << "\noperation=" << operation
           << "\nreproduce_operations=" << reproduce_operations
           << "\nminimization=first-failing-prefix\nmessage=" << message << '\n';
  }
  return 1;
}

int Run(const Options& options) {
  constexpr uint32_t kSequenceCapacity = 64;
  constexpr uint64_t kKvCapacity = 4096;
  inferx::scheduler::ResourceAccountant accountant{inferx::SequenceCount(kSequenceCapacity),
                                                   inferx::KvTokenCount(kKvCapacity)};
  std::map<inferx::ReservationId, inferx::scheduler::ResourceCost> live;
  std::mt19937_64 random(options.seed);
  LifecycleGenerator lifecycle(options.seed ^ 0x6c6966656379636cULL);
  uint64_t expected_sequences = 0;
  uint64_t expected_kv = 0;

  for (uint64_t operation = 0; operation < options.operations; ++operation) {
    const bool reserve = live.empty() || random() % 100 < 60;
    if (reserve) {
      const inferx::scheduler::ResourceCost cost{
          inferx::SequenceCount(1), inferx::TokenCount(static_cast<uint32_t>(random() % 96 + 1))};
      const bool should_fit = expected_sequences + 1 <= kSequenceCapacity &&
                              expected_kv + cost.kv_tokens.value() <= kKvCapacity;
      auto transaction = accountant.BeginReservation(inferx::RequestId(operation + 1), cost);
      if (transaction.ok() != should_fit) {
        return Fail(options, operation, "reservation decision differs from reference");
      }
      if (transaction.ok()) {
        const inferx::ReservationId id = transaction->Commit();
        live.emplace(id, cost);
        ++expected_sequences;
        expected_kv += cost.kv_tokens.value();
      }
    } else {
      auto selected = live.begin();
      std::advance(selected, static_cast<int64_t>(random() % live.size()));
      const inferx::ReservationId id = selected->first;
      const inferx::scheduler::ResourceCost cost = selected->second;
      if (!accountant.Release(id).ok()) {
        return Fail(options, operation, "known reservation release failed");
      }
      live.erase(selected);
      --expected_sequences;
      expected_kv -= cost.kv_tokens.value();
      if (accountant.Release(id).ok()) {
        return Fail(options, operation, "duplicate release unexpectedly succeeded");
      }
    }
    const inferx::scheduler::ResourceSnapshot snapshot = accountant.Snapshot();
    if (!accountant.Validate().ok() || snapshot.sequences_used.value() != expected_sequences ||
        snapshot.kv_tokens_used.value() != expected_kv) {
      return Fail(options, operation, "resource invariant mismatch");
    }
    const inferx::TokenId token =
        inferx::simulator::FakeToken(inferx::RequestId(operation), inferx::TokenOffset(0));
    if (token.value() < 0 || token.value() >= 32000) {
      return Fail(options, operation, "fake token escaped vocabulary");
    }
    if (const char* error = lifecycle.Step(random, operation); error != nullptr) {
      return Fail(options, operation, error);
    }
  }
  for (const auto& [id, cost] : live) {
    static_cast<void>(cost);
    if (!accountant.Release(id).ok()) {
      return Fail(options, options.operations, "final release failed");
    }
  }
  const auto final = accountant.Snapshot();
  if (final.sequences_used.value() != 0 || final.kv_tokens_used.value() != 0 ||
      !accountant.Validate().ok()) {
    return Fail(options, options.operations, "final resources are not zero");
  }
  if (const char* error = lifecycle.Finalize(options.operations); error != nullptr) {
    return Fail(options, options.operations, error);
  }
  if (const char* error = lifecycle.CheckCoverage(options.operations); error != nullptr) {
    return Fail(options, options.operations, error);
  }
  std::printf("seed=0x%llx operations=%llu", static_cast<unsigned long long>(options.seed),
              static_cast<unsigned long long>(options.operations));
  lifecycle.PrintCoverage();
  std::printf(" status=ok\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::optional<Options> options = ParseOptions(argc, argv);
  if (!options) {
    std::fprintf(stderr,
                 "usage: inferx_simulator_stress --seed=<integer> --operations=<positive> "
                 "--failure_output=<directory>\n");
    return 2;
  }
  return Run(*options);
}
