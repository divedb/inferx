#include "inferx/simulator/invariant_checker.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"
#include "inferx/lifecycle/request_state.h"

namespace inferx::simulator {
namespace {

absl::Status Violation(absl::string_view detail) {
  return WithErrorReason(absl::InternalError(absl::StrCat("simulator.invariant: ", detail)),
                         ErrorReason::kInvariantViolation);
}

bool RequiresReservation(RequestState state) {
  return state == RequestState::kPrefillReady || state == RequestState::kPrefilling ||
         state == RequestState::kDecodeReady || state == RequestState::kDecoding ||
         state == RequestState::kCancelling || state == RequestState::kFinishing;
}

}  // namespace

InvariantChecker::InvariantChecker(const RequestRegistry& registry,
                                   const scheduler::ResourceAccountant& resources,
                                   const scheduler::StepPlanPool& plan_pool,
                                   const FakeExecutor& executor,
                                   const std::vector<ResponseEvent>& responses,
                                   TokenCount model_context) noexcept
    : registry_(registry),
      resources_(resources),
      plan_pool_(plan_pool),
      executor_(executor),
      responses_(responses),
      model_context_(model_context) {}

absl::Status InvariantChecker::Validate() const {
  absl::Status resources_valid = resources_.Validate();
  if (!resources_valid.ok()) return resources_valid;
  absl::Status pool_valid = plan_pool_.Validate();
  if (!pool_valid.ok()) return pool_valid;
  absl::Status executor_valid = executor_.Validate();
  if (!executor_valid.ok()) return executor_valid;

  std::vector<const RequestContext*> order;
  registry_.AppendArrivalOrder(order);
  if (order.size() != registry_.size()) {
    return Violation("registry lookup and arrival order sizes differ");
  }
  uint32_t expected_sequences = 0;
  uint64_t expected_kv = 0;
  for (const RequestContext* request : order) {
    if (request == nullptr || request->request == nullptr) {
      return Violation("registry contains a null request");
    }
    if (request->prompt_tokens.size() > UINT32_MAX ||
        request->output_tokens.size() != request->num_committed_output_tokens) {
      return Violation("prompt/output vector counters disagree");
    }
    absl::StatusOr<uint32_t> context_tokens =
        CheckedAdd(static_cast<uint32_t>(request->prompt_tokens.size()),
                   request->num_committed_output_tokens, "simulator.invariant.context_tokens");
    if (!context_tokens.ok() || *context_tokens > model_context_.value()) {
      return Violation("request exceeds model context");
    }
    const bool has_work =
        request->in_flight_step.has_value() && request->in_flight_ticket.has_value() &&
        request->submitted_ticket.has_value() && request->in_flight_work.has_value() &&
        request->in_flight_range.has_value();
    if ((request->num_scheduled_tokens > 0) != has_work || IsInFlight(request->state) != has_work) {
      return Violation("in-flight state, ticket, and scheduled count disagree");
    }
    if (RequiresReservation(request->state) != request->reservation.has_value()) {
      return Violation("request state and reservation ownership disagree");
    }
    if (IsTerminal(request->state) != request->terminal_emitted ||
        IsTerminal(request->state) != request->terminal.has_value()) {
      return Violation("terminal state and durable terminal marker disagree");
    }
    if (request->reservation.has_value()) {
      const std::optional<scheduler::ResourceCost> cost =
          resources_.LookupCost(*request->reservation);
      if (!cost.has_value()) return Violation("request owns an unknown reservation");
      absl::StatusOr<uint32_t> next_sequences =
          CheckedAdd(expected_sequences, cost->sequences.value(), "simulator.invariant.sequences");
      absl::StatusOr<uint64_t> next_kv =
          CheckedAdd(expected_kv, static_cast<uint64_t>(cost->kv_tokens.value()),
                     "simulator.invariant.kv_tokens");
      if (!next_sequences.ok() || !next_kv.ok()) {
        return Violation("reservation sum overflow");
      }
      expected_sequences = *next_sequences;
      expected_kv = *next_kv;
    }

    size_t terminal_responses = 0;
    size_t deltas = 0;
    for (const ResponseEvent& response : responses_) {
      if (const auto* delta = std::get_if<TokenDelta>(&response)) {
        if (delta->request == request->request->id) ++deltas;
      } else if (std::get<TerminalResponse>(response).request == request->request->id) {
        ++terminal_responses;
      }
    }
    if (deltas != request->num_committed_output_tokens ||
        terminal_responses != (request->terminal_emitted ? 1U : 0U)) {
      return Violation("response counts disagree with request counters");
    }
  }
  const scheduler::ResourceSnapshot snapshot = resources_.Snapshot();
  if (expected_sequences != snapshot.sequences_used.value() ||
      expected_kv != snapshot.kv_tokens_used.value()) {
    return Violation("registry reservation sum differs from accountant");
  }
  return absl::OkStatus();
}

absl::Status InvariantChecker::ValidateFinal() const {
  absl::Status valid = Validate();
  if (!valid.ok()) return valid;
  std::vector<const RequestContext*> order;
  registry_.AppendArrivalOrder(order);
  for (const RequestContext* request : order) {
    if (!IsTerminal(request->state)) {
      return Violation("final request is not terminal");
    }
  }
  const scheduler::ResourceSnapshot snapshot = resources_.Snapshot();
  if (snapshot.sequences_used.value() != 0 || snapshot.kv_tokens_used.value() != 0 ||
      resources_.live_reservations() != 0 || executor_.live_tickets() != 0 ||
      plan_pool_.leased() != 0) {
    return Violation("final resource, ticket, or plan ownership is nonzero");
  }
  return absl::OkStatus();
}

}  // namespace inferx::simulator
