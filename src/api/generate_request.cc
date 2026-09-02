#include "inferx/api/generate_request.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"

namespace inferx {

namespace {

absl::Status Invalid(absl::string_view field, absl::string_view message) {
  return FieldError(absl::StatusCode::kInvalidArgument, "request", field, message);
}

}  // namespace

absl::StatusOr<GenerateRequest> ValidateGenerateRequest(const GenerateRequest& request,
                                                        const ModelRecord& model,
                                                        MonotonicTime now) {
  if (request.model != model.id) {
    return Invalid("model", "unknown model (only model 0 exists)");
  }
  if (request.priority != kDefaultPriority) {
    return Invalid("priority", "non-default priority is unimplemented");
  }
  if (request.tenant != TenantScope(0)) {
    return Invalid("tenant", "only the process-default scope is accepted");
  }
  if (request.deadline.has_value() && !(*request.deadline > now)) {
    return Invalid("deadline", "deadline must be strictly in the future");
  }

  const std::vector<TokenId>* tokens = std::get_if<std::vector<TokenId>>(&request.input);
  if (tokens == nullptr) {
    return absl::Status(absl::StatusCode::kUnimplemented,
                        "request.input: text input requires tokenization");
  }
  if (tokens->empty()) {
    return Invalid("input", "prompt must contain at least one token");
  }

  // prompt + output must fit the model context, with checked arithmetic.
  absl::StatusOr<uint64_t> required = CheckedAdd(
      static_cast<uint64_t>(tokens->size()),
      static_cast<uint64_t>(request.generation.max_output_tokens.value()), "request.context");
  if (!required.ok()) {
    return required.status();
  }
  if (*required > model.max_context_tokens.value()) {
    return Invalid("context",
                   absl::StrCat("prompt+output (", *required, ") exceeds model context (",
                                model.max_context_tokens.value(), ")"));
  }
  if (*required > UINT32_MAX) {
    return Invalid("context", "prompt+output exceeds engine token limits");
  }

  return request;
}

}  // namespace inferx
