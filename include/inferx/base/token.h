// Unit-bearing token/count values (ADR 0008).
//
// These are not arithmetic types: calculations unwrap with checked helpers and
// reconstruct the intended unit. Tags keep distinct units non-interchangeable
// even when the storage matches.

#ifndef INFERX_BASE_TOKEN_H_
#define INFERX_BASE_TOKEN_H_

#include <cstdint>
#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace inferx {

struct TokenCountTag {};
struct TokenOffsetTag {};
struct SequenceCountTag {};
struct QueueCapacityTag {};
struct KvTokenCountTag {};
struct ByteCountTag {};

// Checked unit value: explicit construction, value access, comparison, and a
// FromUint64 factory for parsed input. No arithmetic, no implicit conversion.
template <typename Tag, typename Rep>
class UnitValue {
 public:
  explicit constexpr UnitValue(Rep value) noexcept : value_(value) {}

  UnitValue() = delete;

  [[nodiscard]] constexpr Rep value() const noexcept { return value_; }

  friend constexpr bool operator==(const UnitValue&, const UnitValue&) = default;
  friend constexpr auto operator<=>(const UnitValue&, const UnitValue&) = default;

  static absl::StatusOr<UnitValue> FromUint64(uint64_t parsed, absl::string_view field) {
    if (parsed > std::numeric_limits<Rep>::max()) {
      return absl::OutOfRangeError(
          absl::StrCat(field, ": value ", parsed, " exceeds unit storage limit"));
    }
    return UnitValue(static_cast<Rep>(parsed));
  }

 private:
  Rep value_;
};

// uint32-backed units.
using TokenCount = UnitValue<TokenCountTag, uint32_t>;
using TokenOffset = UnitValue<TokenOffsetTag, uint32_t>;
using SequenceCount = UnitValue<SequenceCountTag, uint32_t>;
using QueueCapacity = UnitValue<QueueCapacityTag, uint32_t>;

// uint64-backed aggregates.
using KvTokenCount = UnitValue<KvTokenCountTag, uint64_t>;
using ByteCount = UnitValue<ByteCountTag, uint64_t>;

// Half-open [begin, end) token interval.
struct TokenRange {
  TokenOffset begin;
  TokenOffset end;  // Exclusive.

  // begin <= end; both are non-negative by representation.
  [[nodiscard]] absl::Status Validate(absl::string_view field = "token_range") const;

  [[nodiscard]] absl::StatusOr<TokenCount> size(absl::string_view field = "token_range") const;
};

inline absl::Status TokenRange::Validate(absl::string_view field) const {
  if (end.value() < begin.value()) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, ": end ", end.value(), " before begin ", begin.value()));
  }
  return absl::OkStatus();
}

inline absl::StatusOr<TokenCount> TokenRange::size(absl::string_view field) const {
  absl::Status validated = Validate(field);
  if (!validated.ok()) {
    return validated;
  }
  return TokenCount(end.value() - begin.value());
}

}  // namespace inferx

#endif  // INFERX_BASE_TOKEN_H_
