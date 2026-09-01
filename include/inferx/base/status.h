// Status conventions (m1.md section 7.1; ADR 0002/0008).
//
// absl::Status/absl::StatusOr are the only cross-module error types. This
// header adds the InferX reason classification (stable numeric values used by
// replay and metrics), payload helpers, and the component/field message
// convention. It deliberately does not wrap or alias Status.

#ifndef INFERX_BASE_STATUS_H_
#define INFERX_BASE_STATUS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace inferx {

// Stable replay/metrics classification. It accompanies a status but never
// replaces the canonical code. Renumbering or reusing a value requires a
// replay-schema major change (m1.md section 7.1). The uint16_t
// representation is fixed by that schema contract.
enum class ErrorReason : uint16_t {  // NOLINT(performance-enum-size)
  kNone = 0,
  kInvalidConfig = 1,
  kInvalidWorkload = 2,
  kInvalidRequest = 3,
  kDuplicateRequest = 4,
  kQueueFull = 5,
  kCapacityExhausted = 6,
  kImpossibleContext = 7,
  kInvalidTransition = 8,
  kStaleCompletion = 9,
  kUnknownResource = 10,
  kExecutorRejected = 11,
  kExecutorFailure = 12,
  kDeadlineExpired = 13,
  kExplicitCancellation = 14,
  kShutdown = 15,
  kReplayMismatch = 16,
  kInvariantViolation = 17,
  kIoFailure = 18,
  kEventLimit = 19,
  kDeadlock = 20,
};

// Payload URL carrying the reason classification on an absl::Status.
inline constexpr absl::string_view kErrorReasonPayloadUrl = "type.inferx.dev/error-reason";

// Lowercase closed set used in payloads and replay records.
[[nodiscard]] inline absl::string_view ErrorReasonToName(ErrorReason reason) {
  switch (reason) {
    case ErrorReason::kNone:
      return "none";
    case ErrorReason::kInvalidConfig:
      return "invalid-config";
    case ErrorReason::kInvalidWorkload:
      return "invalid-workload";
    case ErrorReason::kInvalidRequest:
      return "invalid-request";
    case ErrorReason::kDuplicateRequest:
      return "duplicate-request";
    case ErrorReason::kQueueFull:
      return "queue-full";
    case ErrorReason::kCapacityExhausted:
      return "capacity-exhausted";
    case ErrorReason::kImpossibleContext:
      return "impossible-context";
    case ErrorReason::kInvalidTransition:
      return "invalid-transition";
    case ErrorReason::kStaleCompletion:
      return "stale-completion";
    case ErrorReason::kUnknownResource:
      return "unknown-resource";
    case ErrorReason::kExecutorRejected:
      return "executor-rejected";
    case ErrorReason::kExecutorFailure:
      return "executor-failure";
    case ErrorReason::kDeadlineExpired:
      return "deadline-expired";
    case ErrorReason::kExplicitCancellation:
      return "explicit-cancellation";
    case ErrorReason::kShutdown:
      return "shutdown";
    case ErrorReason::kReplayMismatch:
      return "replay-mismatch";
    case ErrorReason::kInvariantViolation:
      return "invariant-violation";
    case ErrorReason::kIoFailure:
      return "io-failure";
    case ErrorReason::kEventLimit:
      return "event-limit";
    case ErrorReason::kDeadlock:
      return "deadlock";
  }
  return "unknown";
}

[[nodiscard]] inline std::optional<ErrorReason> ErrorReasonFromName(absl::string_view name) {
  for (uint32_t value = 0; value <= 20; ++value) {
    const auto reason = static_cast<ErrorReason>(value);
    if (ErrorReasonToName(reason) == name) {
      return reason;
    }
  }
  return std::nullopt;
}

// Attaches the reason payload (returns a copy; the input is unchanged).
[[nodiscard]] inline absl::Status WithErrorReason(absl::Status status, ErrorReason reason) {
  status.SetPayload(kErrorReasonPayloadUrl, absl::Cord(ErrorReasonToName(reason)));
  return status;
}

inline absl::StatusOr<ErrorReason> GetErrorReason(const absl::Status& status) {
  if (status.ok()) {
    return ErrorReason::kNone;
  }
  std::optional<absl::Cord> payload = status.GetPayload(kErrorReasonPayloadUrl);
  if (!payload.has_value()) {
    return absl::InvalidArgumentError(
        "status.reason: error status carries no error-reason payload");
  }
  const std::string name(payload->Flatten());
  std::optional<ErrorReason> reason = ErrorReasonFromName(name);
  if (!reason.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("status.reason: unknown error-reason payload '", name, "'"));
  }
  if (*reason == ErrorReason::kNone) {
    return absl::InvalidArgumentError("status.reason: error status carries reason 'none'");
  }
  return *reason;
}

// Message convention: "<component>.<field>: <message>", e.g.
// "config.max_active_sequences: must be <= max_queued_requests".
[[nodiscard]] absl::Status FieldError(absl::StatusCode code, absl::string_view component,
                                      absl::string_view field, absl::string_view message);
[[nodiscard]] absl::Status ComponentError(absl::StatusCode code, absl::string_view component,
                                          absl::string_view message);

inline absl::Status FieldError(absl::StatusCode code, absl::string_view component,
                               absl::string_view field, absl::string_view message) {
  return absl::Status(code, absl::StrCat(component, ".", field, ": ", message));
}

inline absl::Status ComponentError(absl::StatusCode code, absl::string_view component,
                                   absl::string_view message) {
  return absl::Status(code, absl::StrCat(component, ": ", message));
}

}  // namespace inferx

#endif  // INFERX_BASE_STATUS_H_
