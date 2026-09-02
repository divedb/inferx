// Execution ticket/completion values. IDs and
// epochs only — never a context pointer or owning view.

#ifndef INFERX_ENGINE_EXECUTION_COMPLETION_H_
#define INFERX_ENGINE_EXECUTION_COMPLETION_H_

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"
#include "inferx/scheduler/work_kind.h"

namespace inferx {

// Hardware-neutral ticket returned for one accepted plan. Request identity is
// carried per completion item because one ticket may own a batch.
struct ExecutionTicket {
  ExecutionTicketId id{0};
  StepId step{0};
  uint32_t item_count = 0;
};

// One completion item echoes every identity/range needed to reject stale or
// wrong-request work without touching newer state.
struct ExecutionCompletion {
  ExecutionTicketId ticket{0};
  StepId step{0};
  uint32_t item_ordinal = 0;
  uint32_t item_count = 0;
  RequestId request{0};
  SequenceId sequence{0};
  RequestEpoch epoch{0};
  WorkKind kind = WorkKind::kPrefill;
  TokenRange scheduled_tokens{TokenOffset(0), TokenOffset(0)};
  absl::Status status;
  ErrorReason error_reason = ErrorReason::kNone;
  std::optional<TokenId> output_token;
};

}  // namespace inferx

#endif  // INFERX_ENGINE_EXECUTION_COMPLETION_H_
