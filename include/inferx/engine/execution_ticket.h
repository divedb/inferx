// Ticket id helpers kept separate from the value contract for include
// hygiene: contexts and events include execution_completion.h directly.
#ifndef INFERX_ENGINE_EXECUTION_TICKET_H_
#define INFERX_ENGINE_EXECUTION_TICKET_H_

#include "inferx/engine/execution_completion.h"

namespace inferx {

// A completion matches submitted work exactly.
[[nodiscard]] bool TicketMatches(const ExecutionTicket& submitted,
                                 const ExecutionTicket& completion);

inline bool TicketMatches(const ExecutionTicket& submitted, const ExecutionTicket& completion) {
  return submitted.id == completion.id && submitted.step == completion.step &&
         submitted.item_count == completion.item_count;
}

[[nodiscard]] inline bool TicketMatches(const ExecutionTicket& submitted,
                                        const ExecutionCompletion& completion) {
  return submitted.id == completion.ticket && submitted.step == completion.step &&
         submitted.item_count == completion.item_count &&
         completion.item_ordinal < completion.item_count;
}

}  // namespace inferx

#endif  // INFERX_ENGINE_EXECUTION_TICKET_H_
