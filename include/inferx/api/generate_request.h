// Immutable generation request values (m1.md section 9.1). Validation lives
// in inferx::api::ValidateGenerateRequest; after validation the request is
// moved into the registry and never mutated.

#ifndef INFERX_API_GENERATE_REQUEST_H_
#define INFERX_API_GENERATE_REQUEST_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"

namespace inferx {

// Non-interchangeable priority: M1 accepts only the default zero.
class Priority {
 public:
  explicit constexpr Priority(int32_t value) noexcept : value_(value) {}

  Priority() = delete;

  [[nodiscard]] constexpr int32_t value() const noexcept { return value_; }

  friend constexpr bool operator==(const Priority&, const Priority&) = default;

 private:
  int32_t value_;
};

inline constexpr Priority kDefaultPriority{0};

using PromptInput = std::variant<std::string, std::vector<TokenId>>;

struct GenerationLimits {
  TokenCount max_output_tokens;
};

struct GenerateRequest {
  RequestId id;
  ModelId model;
  PromptInput input;
  GenerationLimits generation;
  Priority priority = kDefaultPriority;
  std::optional<MonotonicTime> deadline;
  TenantScope tenant{0};  // explicitly constructed process-default scope
};

// The single M1 fake model capability record (m1.md section 9.1): ModelId(0),
// context 32,768, no text support. M3 replaces this with real artifacts.
struct ModelRecord {
  ModelId id{0};
  TokenCount max_context_tokens{32768};
  bool accepts_text = false;
};

// Validation rules (m1.md section 9.1): token input only (text is
// Unimplemented in M1); nonempty prompt; prompt + output must fit the model
// context and the configured token budget; output limit must not overflow
// the configured cap; deadline must not already be expired; priority must be
// the default; tenant must be the process default; model must be the known
// fake model. The clock is passed in for expiry checks (manual time in the
// simulator).
absl::StatusOr<GenerateRequest> ValidateGenerateRequest(const GenerateRequest& request,
                                                        const ModelRecord& model,
                                                        MonotonicTime now);

}  // namespace inferx

#endif  // INFERX_API_GENERATE_REQUEST_H_
