// Unit-value, range, checked-arithmetic, and clock boundary tests
// (m1.md section 17.1 base).
#include <gtest/gtest.h>

#include <limits>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/clock.h"
#include "inferx/base/status_macros.h"
#include "inferx/base/token.h"

namespace {

using inferx::ByteCount;
using inferx::CheckedAdd;
using inferx::CheckedCeilDiv;
using inferx::CheckedMul;
using inferx::CheckedNarrow;
using inferx::CheckedSub;
using inferx::KvTokenCount;
using inferx::ManualClock;
using inferx::MonotonicTime;
using inferx::Nanoseconds;
using inferx::QueueCapacity;
using inferx::SequenceCount;
using inferx::TokenCount;
using inferx::TokenOffset;
using inferx::TokenRange;

TEST(UnitValueTest, FromUint64AcceptsBoundary) {
  ASSERT_TRUE(TokenCount::FromUint64(0, "t").ok());
  EXPECT_EQ(TokenCount::FromUint64(UINT32_MAX, "t").value(), TokenCount(UINT32_MAX));
  EXPECT_EQ(KvTokenCount::FromUint64(UINT64_MAX, "kv").value(), KvTokenCount(UINT64_MAX));
}

TEST(UnitValueTest, FromUint64RejectsOutOfRange) {
  const absl::StatusOr<TokenCount> rejected = TokenCount::FromUint64(uint64_t(UINT32_MAX) + 1, "t");
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_NE(rejected.status().message().find("t:"), std::string::npos);
}

TEST(TokenRangeTest, ValidateAndSize) {
  const TokenRange range{TokenOffset(3), TokenOffset(9)};
  ASSERT_TRUE(range.Validate().ok());
  EXPECT_EQ(range.size().value(), TokenCount(6));
  const TokenRange empty_range{TokenOffset(5), TokenOffset(5)};
  EXPECT_EQ(empty_range.size().value(), TokenCount(0));
  const TokenRange inverted{TokenOffset(9), TokenOffset(3)};
  EXPECT_EQ(inverted.Validate().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(inverted.size().ok());
}

TEST(CheckedMathTest, AddBoundaries) {
  EXPECT_EQ(CheckedAdd(1, 2).value(), 3);
  EXPECT_EQ(CheckedAdd(uint32_t(0), uint32_t(0)).value(), 0u);
  ASSERT_FALSE(CheckedAdd(INT32_MAX, 1).ok());
  ASSERT_FALSE(CheckedAdd(INT32_MIN, -1).ok());
  ASSERT_FALSE(CheckedAdd(uint32_t(UINT32_MAX), uint32_t(1)).ok());
  EXPECT_EQ(CheckedAdd(-5, 3).value(), -2);
}

TEST(CheckedMathTest, SubBoundaries) {
  EXPECT_EQ(CheckedSub(10, 4).value(), 6);
  ASSERT_FALSE(CheckedSub(INT32_MIN, 1).ok());
  ASSERT_FALSE(CheckedSub(INT32_MAX, -1).ok());
  ASSERT_FALSE(CheckedSub(uint32_t(1), uint32_t(2)).ok());
}

TEST(CheckedMathTest, MulBoundaries) {
  EXPECT_EQ(CheckedMul(6, 7).value(), 42);
  ASSERT_FALSE(CheckedMul(INT32_MAX, 2).ok());
  ASSERT_FALSE(CheckedMul(INT32_MIN, -1).ok());
  ASSERT_FALSE(CheckedMul(-2, INT32_MIN).ok());
  ASSERT_FALSE(CheckedMul(uint32_t(UINT32_MAX), uint32_t(2)).ok());
  EXPECT_EQ(CheckedMul(0, INT32_MIN).value(), 0);
}

TEST(CheckedMathTest, CeilDiv) {
  EXPECT_EQ(CheckedCeilDiv(7, 2).value(), 4);
  EXPECT_EQ(CheckedCeilDiv(8, 2).value(), 4);
  EXPECT_EQ(CheckedCeilDiv(0, 5).value(), 0);
  EXPECT_EQ(CheckedCeilDiv(-7, 2).value(), -3);
  EXPECT_EQ(CheckedCeilDiv(7, -2).value(), -3);
  EXPECT_EQ(CheckedCeilDiv(-7, -2).value(), 4);
  ASSERT_FALSE(CheckedCeilDiv(1, 0).ok());
  ASSERT_FALSE(CheckedCeilDiv(INT32_MIN, -1).ok());
}

TEST(CheckedMathTest, NarrowSignCombinations) {
  EXPECT_EQ(CheckedNarrow<int8_t>(int32_t(100)).value(), int8_t(100));
  ASSERT_FALSE(CheckedNarrow<int8_t>(int32_t(200)).ok());
  ASSERT_FALSE(CheckedNarrow<uint8_t>(int32_t(-1)).ok());
  ASSERT_FALSE(CheckedNarrow<uint16_t>(uint32_t(70000)).ok());
  EXPECT_EQ(CheckedNarrow<int64_t>(int32_t(-5)).value(), int64_t(-5));
  ASSERT_FALSE(CheckedNarrow<int32_t>(uint32_t(3000000000u)).ok());
}

TEST(CheckedMathTest, ByteSizeRoutesThroughCheckedMul) {
  EXPECT_EQ(inferx::CheckedByteSize(4, 8).value().value(), 32u);
  const absl::StatusOr<ByteCount> overflow = inferx::CheckedByteSize(UINT64_MAX, 2, "tensor");
  ASSERT_FALSE(overflow.ok());
  EXPECT_EQ(overflow.status().code(), absl::StatusCode::kOutOfRange);
}

TEST(CheckedMathTest, ContextAppearsInMessage) {
  const absl::StatusOr<int32_t> failed = CheckedAdd(INT32_MAX, 1, "config.field");
  ASSERT_FALSE(failed.ok());
  EXPECT_NE(failed.status().message().find("config.field"), std::string::npos);
}

TEST(ManualClockTest, MonotonicForwardOnly) {
  ManualClock clock(MonotonicTime{} + Nanoseconds(100));
  EXPECT_EQ(clock.Now().time_since_epoch().count(), 100);
  ASSERT_TRUE(clock.Advance(Nanoseconds(50)).ok());
  EXPECT_EQ(clock.Now().time_since_epoch().count(), 150);
  EXPECT_FALSE(clock.Advance(Nanoseconds(-1)).ok());
}

TEST(ManualClockTest, OverflowRejected) {
  ManualClock clock(MonotonicTime::max() - Nanoseconds(10));
  ASSERT_TRUE(clock.Advance(Nanoseconds(5)).ok());
  ASSERT_FALSE(clock.Advance(Nanoseconds(100)).ok());
  // Failed advance did not move time.
  EXPECT_EQ(clock.Now().time_since_epoch().count(),
            (MonotonicTime::max() - Nanoseconds(5)).time_since_epoch().count());
}

TEST(ManualClockTest, DeadlineBoundaryIsInclusive) {
  ManualClock clock(MonotonicTime{} + Nanoseconds(1000));
  const auto deadline = clock.Now();     // deadline == now
  EXPECT_TRUE(clock.Now() >= deadline);  // expires when Now() >= deadline
}

TEST(StatusMacroTest, AssignOrReturnMovesUniquePtr) {
  const auto run = []() -> absl::Status {
    auto make = []() -> absl::StatusOr<std::unique_ptr<int>> { return std::make_unique<int>(7); };
    std::unique_ptr<int> value;
    INFERX_ASSIGN_OR_RETURN(value, make());
    EXPECT_EQ(*value, 7);
    return absl::OkStatus();
  };
  EXPECT_TRUE(run().ok());
}

}  // namespace
