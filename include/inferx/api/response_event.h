// Response events (m1.md section 9.2): per-request token deltas precede
// exactly one terminal response. M1 exposes no text/logprobs/usage metadata.

#ifndef INFERX_API_RESPONSE_EVENT_H_
#define INFERX_API_RESPONSE_EVENT_H_

#include <cstdint>
#include <optional>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"

namespace inferx {

struct TokenDelta {
  RequestId request;
  TokenId token;
  TokenOffset output_position;
};

enum class FinishReason : uint8_t {
  kLength,
  kSimulatedEos,
  kCancelled,
  kDeadline,
  kExecutorError,
  kShutdown,
};

[[nodiscard]] absl::string_view ToString(FinishReason reason);
[[nodiscard]] std::optional<FinishReason> FinishReasonFromName(absl::string_view name);

struct TerminalResponse {
  // Zero-initialized identity placeholder; real records are always built
  // with the actual request id (controller/tests).
  RequestId request{0};
  FinishReason reason = FinishReason::kLength;
  absl::StatusCode status = absl::StatusCode::kOk;
  std::optional<ErrorReason> error_reason;  // nullopt on success
  TokenCount prompt_tokens{0};
  TokenCount output_tokens{0};
  MonotonicTime terminal_time{};
};

using ResponseEvent = std::variant<TokenDelta, TerminalResponse>;

}  // namespace inferx

#endif  // INFERX_API_RESPONSE_EVENT_H_
