// StrongId boundary tests (m1.md section 17.1 base).
#include <gtest/gtest.h>

#include <optional>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/id.h"

namespace {

using inferx::BlockId;
using inferx::DeviceId;
using inferx::IdGenerator;
using inferx::Rank;
using inferx::RequestId;
using inferx::TokenId;

TEST(StrongIdTest, ZeroAndMaxRoundTrip) {
  // Zero is valid for ranks/devices/tokens: no invalid sentinel exists.
  EXPECT_EQ(Rank(0).value(), 0u);
  EXPECT_EQ(DeviceId(0).value(), 0u);
  EXPECT_EQ(TokenId(0).value(), 0);
  EXPECT_EQ(RequestId(UINT64_MAX).value(), UINT64_MAX);
  EXPECT_EQ(BlockId(UINT32_MAX).value(), UINT32_MAX);
  EXPECT_EQ(TokenId(-1).value(), -1);  // negative token IDs are representable
}

TEST(StrongIdTest, ComparisonAndHash) {
  const RequestId first(7);
  const RequestId second(7);
  const RequestId third(8);
  EXPECT_EQ(first, second);
  EXPECT_NE(first, third);
  EXPECT_LT(first, third);
  const size_t hash_a = absl::HashOf(first);
  const size_t hash_b = absl::HashOf(second);
  EXPECT_EQ(hash_a, hash_b);

  // Same representation, different tag: distinct identity types.
  const BlockId block(7);
  const size_t hash_block = absl::HashOf(block);
  (void)hash_block;  // compiles independently of RequestId's hash space
}

TEST(StrongIdTest, Formatting) {
  EXPECT_EQ(absl::StrCat(RequestId(42)), "42");
  EXPECT_EQ(absl::StrCat(TokenId(-3)), "-3");
}

TEST(StrongIdTest, OptionalAbsence) {
  const std::optional<RequestId> absent;
  EXPECT_FALSE(absent.has_value());
  const std::optional<RequestId> present(RequestId(1));
  ASSERT_TRUE(present.has_value());
  EXPECT_EQ(*present, RequestId(1));
}

TEST(IdGeneratorTest, SequentialFromExplicitInitial) {
  IdGenerator<RequestId> generator(RequestId(1));
  EXPECT_EQ(generator.Next("simulator").value(), RequestId(1));
  EXPECT_EQ(generator.Next("simulator").value(), RequestId(2));
}

TEST(IdGeneratorTest, OverflowFailsWithoutAdvancing) {
  IdGenerator<inferx::RequestEpoch> epoch(inferx::RequestEpoch(UINT32_MAX));
  absl::StatusOr<inferx::RequestEpoch> rejected = epoch.Next("simulator");
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kOutOfRange);
  // Still exhausted; no wraparound to zero.
  EXPECT_FALSE(epoch.Next("simulator").ok());
}

}  // namespace
