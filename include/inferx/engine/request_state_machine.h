// Declarative request transition table and pure decision function. One
// constexpr rule array (wildcards expanded to
// concrete state/event rows); absent cells are FailedPrecondition/kInvalidTransition
// with no side effects. Conditional destinations are named pure resolvers.

#ifndef INFERX_ENGINE_REQUEST_STATE_MACHINE_H_
#define INFERX_ENGINE_REQUEST_STATE_MACHINE_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "inferx/api/response_event.h"
#include "inferx/engine/request_context.h"
#include "inferx/engine/request_event.h"
#include "inferx/lifecycle/request_state.h"

namespace inferx {

// Side effects a transition instructs the coordinator to secure/release
// around the no-fail commit.
enum class TransitionEffect : uint8_t {
  kNone = 0,
  kReleaseReservation = 1u << 0,
  kIncrementEpoch = 1u << 1,
  kClearInFlight = 1u << 2,
};

[[nodiscard]] constexpr TransitionEffect operator|(TransitionEffect a, TransitionEffect b) {
  return static_cast<TransitionEffect>(static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
}

[[nodiscard]] constexpr bool operator&(TransitionEffect a, TransitionEffect b) {
  return (static_cast<uint16_t>(a) & static_cast<uint16_t>(b)) != 0;
}

// Terminal bookkeeping decided alongside the transition.
struct TerminalOutcome {
  FinishReason reason = FinishReason::kLength;
  absl::StatusCode status = absl::StatusCode::kOk;
  std::optional<ErrorReason> error_reason;
};

struct TransitionDecision {
  RequestState next;
  TransitionEffect effects = TransitionEffect::kNone;
  std::optional<TerminalOutcome> terminal;  // set when entering a terminal state
};

// Pure decision: reads the declarative table plus explicit guards
// (matching step/ticket/epoch, output counters, payload consistency).
// Invalid/absent cells return FailedPrecondition and change nothing.
absl::StatusOr<TransitionDecision> DecideTransition(const RequestContext& request,
                                                    const RequestEvent& event);

// One concrete table row. Wildcard rows are expanded here; a unit test
// rejects duplicate cells.
struct TransitionRule {
  RequestState current;
  RequestEventKind event;
  // Resolves the next state; conditional rows read context counters AND the
  // event payload (a completion's own committed outputs are only in the
  // event, since the context updates at commit).
  RequestState (*resolve_next)(const RequestContext&, const RequestEvent&) = nullptr;
  RequestState fixed_next;  // used when resolve_next is null
  TransitionEffect effects = TransitionEffect::kNone;
  bool terminal = false;
};

// The expanded table (compile-time constant; the only transition authority).
// Wildcard rows are concrete here.
struct TransitionRuleSpan {
  const TransitionRule* data;
  size_t size;
};

[[nodiscard]] TransitionRuleSpan TransitionRules();

}  // namespace inferx

#endif  // INFERX_ENGINE_REQUEST_STATE_MACHINE_H_
