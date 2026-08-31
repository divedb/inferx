// Ticket id helpers kept separate from the value contract for include
// hygiene: contexts and events include execution_completion.h directly.
#ifndef INFERX_ENGINE_EXECUTION_TICKET_H_
#define INFERX_ENGINE_EXECUTION_TICKET_H_

#include "inferx/engine/execution_completion.h"

namespace inferx {

// A completion matches submitted work exactly (m1.md section 11.4).
[[nodiscard]] bool TicketMatches(const ExecutionTicket& submitted,
                                 const ExecutionTicket& completion);

inline bool TicketMatches(const ExecutionTicket& submitted, const ExecutionTicket& completion) {
  return submitted.request == completion.request && submitted.sequence == completion.sequence &&
         submitted.epoch == completion.epoch && submitted.step == completion.step &&
         submitted.ticket == completion.ticket &&
         submitted.scheduled_tokens == completion.scheduled_tokens;
}

}  // namespace inferx

#endif  // INFERX_ENGINE_EXECUTION_TICKET_H_
