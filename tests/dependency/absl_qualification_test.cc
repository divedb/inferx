// Abseil qualification test (ADR 0002): constructs and inspects
// absl::Status/StatusOr entirely inside this test target so the error API is
// proven without exposing Abseil from inferx::base prematurely.
#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace {

absl::StatusOr<int> QualifiedValue(int value) {
  if (value < 0) {
    return absl::InvalidArgumentError("negative values are not qualified");
  }
  return value;
}

TEST(AbseilQualificationTest, StatusConstructionAndCodes) {
  const absl::Status ok = absl::OkStatus();
  ASSERT_TRUE(ok.ok());
  EXPECT_EQ(ok.code(), absl::StatusCode::kOk);

  const absl::Status resource = absl::ResourceExhaustedError("kv pool");
  EXPECT_EQ(resource.code(), absl::StatusCode::kResourceExhausted);
  EXPECT_NE(resource.message().find("kv pool"), std::string::npos);
}

TEST(AbseilQualificationTest, StatusOrPropagation) {
  const absl::StatusOr<int> good = QualifiedValue(7);
  ASSERT_TRUE(good.ok());
  EXPECT_EQ(*good, 7);

  const absl::StatusOr<int> bad = QualifiedValue(-1);
  ASSERT_FALSE(bad.ok());
  EXPECT_EQ(bad.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(AbseilQualificationTest, MessageHelpers) {
  const std::string composed = absl::StrCat("component=", "base", " retryable=", true);
  // absl::AlphaNum formats bool as 1/0.
  EXPECT_EQ(composed, "component=base retryable=1");

  const absl::Status status = absl::UnavailableError(composed);
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
}

}  // namespace
