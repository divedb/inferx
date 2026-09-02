// Request context: mutable engine state owned by the registry, mutated only
// by RequestController. Never part of the public
// request API; plans and completions carry IDs/epochs, never pointers here.

#ifndef INFERX_ENGINE_REQUEST_CONTEXT_H_
#define INFERX_ENGINE_REQUEST_CONTEXT_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "inferx/api/generate_request.h"
#include "inferx/api/response_event.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/engine/execution_completion.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/scheduler/work_kind.h"

namespace inferx {

// Deterministic live order: arrival time, then ordinal, then request ID.
struct ArrivalKey {
  MonotonicTime arrival_time;
  uint64_t arrival_ordinal = 0;
  RequestId request;

  friend bool operator<(const ArrivalKey& a, const ArrivalKey& b) {
    if (a.arrival_time != b.arrival_time) {
      return a.arrival_time < b.arrival_time;
    }
    if (a.arrival_ordinal != b.arrival_ordinal) {
      return a.arrival_ordinal < b.arrival_ordinal;
    }
    return a.request < b.request;
  }
};

struct RequestContext {
  // Immutable after admission (const access through the owning pointer).
  std::unique_ptr<const GenerateRequest> request;

  SequenceId sequence{0};
  RequestEpoch epoch{0};
  ArrivalKey arrival{MonotonicTime{}, 0, RequestId(0)};  // assigned by Insert

  RequestState state = RequestState::kReceived;
  std::optional<TerminalResponse> pending_terminal_reason;  // cancellation cause

  std::vector<TokenId> prompt_tokens;
  std::vector<TokenId> output_tokens;  // committed synthetic outputs

  uint32_t num_computed_tokens = 0;
  uint32_t num_scheduled_tokens = 0;  // outstanding work; zero on every exit
  uint32_t num_committed_output_tokens = 0;

  // Conservative admission cost may exceed an internal synthetic finish
  // guard; normally equal to request.generation.max_output_tokens.
  TokenCount reservation_output_tokens{0};

  std::optional<ReservationId> reservation;
  std::optional<StepId> in_flight_step;
  std::optional<ExecutionTicketId> in_flight_ticket;
  std::optional<ExecutionTicket> submitted_ticket;
  std::optional<WorkKind> in_flight_work;
  std::optional<TokenRange> in_flight_range;

  std::optional<TerminalResponse> terminal;
  bool terminal_emitted = false;

  [[nodiscard]] bool OutputRemains() const {
    return num_committed_output_tokens < request->generation.max_output_tokens.value();
  }
};

}  // namespace inferx

#endif  // INFERX_ENGINE_REQUEST_CONTEXT_H_
