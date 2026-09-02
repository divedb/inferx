// Status convention and macro behavior tests.
#include "absl/status/status.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/status.h"
#include "inferx/base/status_macros.h"

namespace {

using inferx::ComponentError;
using inferx::ErrorReason;
using inferx::ErrorReasonFromName;
using inferx::ErrorReasonToName;
using inferx::FieldError;
using inferx::GetErrorReason;
using inferx::WithErrorReason;

TEST(ErrorReasonTest, NamesRoundTrip) {
  for (uint16_t value = 0; value <= 20; ++value) {
    const auto reason = static_cast<ErrorReason>(value);
    const absl::string_view name = ErrorReasonToName(reason);
    EXPECT_FALSE(name.empty());
    const auto parsed = ErrorReasonFromName(name);
    if (!parsed.has_value()) {
      FAIL() << "no round trip for " << name;
      return;
    }
    EXPECT_EQ(*parsed, reason);
  }
}

TEST(ErrorReasonTest, UnknownNameRejected) {
  EXPECT_FALSE(ErrorReasonFromName("garbage").has_value());
  EXPECT_FALSE(ErrorReasonFromName("").has_value());
}

TEST(ErrorReasonPayloadTest, RoundTripPreservesReason) {
  const absl::Status base = FieldError(absl::StatusCode::kInvalidArgument, "config",
                                       "max_active_sequences", "must be <= max_queued_requests");
  const absl::Status with_reason = WithErrorReason(base, ErrorReason::kInvalidConfig);
  // Payload attachment copies; the original stays payload-free.
  EXPECT_FALSE(base.GetPayload(inferx::kErrorReasonPayloadUrl).has_value());

  const absl::StatusOr<ErrorReason> reason = GetErrorReason(with_reason);
  ASSERT_TRUE(reason.ok());
  EXPECT_EQ(*reason, ErrorReason::kInvalidConfig);
}

TEST(ErrorReasonPayloadTest, OkStatusHasNone) {
  const absl::StatusOr<ErrorReason> reason = GetErrorReason(absl::OkStatus());
  ASSERT_TRUE(reason.ok());
  EXPECT_EQ(*reason, ErrorReason::kNone);
}

TEST(ErrorReasonPayloadTest, MissingPayloadRejected) {
  const absl::StatusOr<ErrorReason> reason = GetErrorReason(absl::InternalError("bare"));
  ASSERT_FALSE(reason.ok());
  EXPECT_EQ(reason.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ErrorReasonPayloadTest, MalformedPayloadRejected) {
  absl::Status status = absl::InternalError("bad payload");
  status.SetPayload(inferx::kErrorReasonPayloadUrl, absl::Cord("not-a-reason"));
  EXPECT_FALSE(GetErrorReason(status).ok());

  absl::Status none_payload = absl::InternalError("none payload");
  none_payload.SetPayload(inferx::kErrorReasonPayloadUrl, absl::Cord("none"));
  EXPECT_FALSE(GetErrorReason(none_payload).ok());
}

TEST(MessageConventionTest, ComponentFieldPrefix) {
  const absl::Status status = FieldError(absl::StatusCode::kFailedPrecondition, "lifecycle",
                                         "request", "terminal state is absorbing");
  EXPECT_EQ(status.message(), "lifecycle.request: terminal state is absorbing");
  const absl::Status component = ComponentError(absl::StatusCode::kUnimplemented, "tokenizer",
                                                "text input requires tokenization");
  EXPECT_EQ(component.message(), "tokenizer: text input requires tokenization");
}

namespace {

int evaluations = 0;

absl::Status EvaluateOnceStatus() {
  ++evaluations;
  return absl::OkStatus();
}

absl::Status EvaluateOnceFailure() {
  ++evaluations;
  return absl::CancelledError("stop");
}

absl::StatusOr<std::unique_ptr<int>> EvaluateOnceValue(int v) {
  ++evaluations;
  return std::make_unique<int>(v);
}

absl::StatusOr<std::unique_ptr<int>> EvaluateOnceFailureOr() {
  ++evaluations;
  return absl::NotFoundError("missing");
}

}  // namespace

TEST(StatusMacrosTest, ReturnIfErrorEvaluatesOnce) {
  evaluations = 0;
  const auto ok_path = []() -> absl::Status {
    INFERX_RETURN_IF_ERROR(EvaluateOnceStatus());
    return absl::OkStatus();
  };
  EXPECT_TRUE(ok_path().ok());
  EXPECT_EQ(evaluations, 1);

  evaluations = 0;
  const auto fail_path = []() -> absl::Status {
    INFERX_RETURN_IF_ERROR(EvaluateOnceFailure());
    ADD_FAILURE() << "unreachable";
    return absl::OkStatus();
  };
  EXPECT_EQ(fail_path().code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(evaluations, 1);
}

TEST(StatusMacrosTest, AssignOrReturnEvaluatesOnceAndMoves) {
  evaluations = 0;
  const auto ok_path = []() -> absl::Status {
    std::unique_ptr<int> value;
    INFERX_ASSIGN_OR_RETURN(value, EvaluateOnceValue(21));
    EXPECT_EQ(*value, 21);
    return absl::OkStatus();
  };
  EXPECT_TRUE(ok_path().ok());
  EXPECT_EQ(evaluations, 1);

  evaluations = 0;
  const auto fail_path = []() -> absl::Status {
    std::unique_ptr<int> value;
    INFERX_ASSIGN_OR_RETURN(value, EvaluateOnceFailureOr());
    ADD_FAILURE() << "unreachable";
    return absl::OkStatus();
  };
  EXPECT_EQ(fail_path().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(evaluations, 1);
}

TEST(StatusMacrosTest, NestedScopesAndPayloadPreservation) {
  const absl::Status inner =
      WithErrorReason(absl::ResourceExhaustedError("queue"), ErrorReason::kQueueFull);
  const auto outer = [&]() -> absl::Status {
    INFERX_RETURN_IF_ERROR([&]() -> absl::Status {
      INFERX_RETURN_IF_ERROR(inner);
      return absl::OkStatus();
    }());
    return absl::OkStatus();
  }();
  EXPECT_EQ(outer.code(), absl::StatusCode::kResourceExhausted);
  const absl::StatusOr<ErrorReason> reason = GetErrorReason(outer);
  ASSERT_TRUE(reason.ok());
  EXPECT_EQ(*reason, ErrorReason::kQueueFull);
}

TEST(StatusMacrosTest, AssignsIntoPreDeclaredLvalue) {
  const auto run = []() -> absl::Status {
    absl::StatusOr<std::string> producer = absl::StrCat("infer", "x");
    std::string assigned;
    INFERX_ASSIGN_OR_RETURN(assigned, producer);
    EXPECT_EQ(assigned, "inferx");
    return absl::OkStatus();
  };
  EXPECT_TRUE(run().ok());
}

}  // namespace
