// Execution ticket/completion values (m1.md sections 11.2, 13). IDs and
// epochs only — never a context pointer or owning view.

#ifndef INFERX_ENGINE_EXECUTION_COMPLETION_H_
#define INFERX_ENGINE_EXECUTION_COMPLETION_H_

#include <optional>

#include "absl/status/status.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"

namespace inferx {

// Identifies submitted work across plan, ticket, and completion.
struct ExecutionTicket {
  RequestId request{0};
  SequenceId sequence{0};
  RequestEpoch epoch{0};
  StepId step{0};
  ExecutionTicketId ticket{0};
  TokenCount scheduled_tokens{0};
};

// Fake-executor completion value (M1 shape; M2 replaces the backend, not the
// value contract).
struct ExecutionCompletion {
  ExecutionTicket ticket;
  bool success = false;
  std::optional<absl::Status> failure;  // set when !success
  // Prompt/decode work marked computed (m1.md section 10.1)...
  TokenCount computed_tokens{0};
  // ...and synthetic output tokens committed by this completion.
  TokenCount committed_tokens{0};
};

}  // namespace inferx

#endif  // INFERX_ENGINE_EXECUTION_COMPLETION_H_
