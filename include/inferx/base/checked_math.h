// Checked integral arithmetic (ADR 0008).
//
// Every parsed or resource-bearing sum/subtraction/product goes through these
// helpers: no silent truncation, no reliance on wraparound, signedness is
// handled explicitly via if-constexpr branches.

#ifndef INFERX_BASE_CHECKED_MATH_H_
#define INFERX_BASE_CHECKED_MATH_H_

#include <concepts>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "inferx/base/token.h"

namespace inferx {

namespace internal {

template <typename T>
concept CheckedInteger = std::integral<T> && !std::is_same_v<T, bool>;

inline absl::Status OverflowError(absl::string_view context, absl::string_view op,
                                  absl::string_view detail) {
  return absl::OutOfRangeError(
      absl::StrCat(context.empty() ? "checked_math" : context, ": ", op, detail));
}

}  // namespace internal

// CheckedAdd/CheckedSub/CheckedMul: same-type operands, contextual result.
template <internal::CheckedInteger T>
absl::StatusOr<T> CheckedAdd(T a, T b, absl::string_view context = {}) {
  if constexpr (std::is_signed_v<T>) {
    if ((b > 0 && a > std::numeric_limits<T>::max() - b) ||
        (b < 0 && a < std::numeric_limits<T>::min() - b)) {
      return internal::OverflowError(context, "add", " overflow (signed)");
    }
  } else {
    if (a > std::numeric_limits<T>::max() - b) {
      return internal::OverflowError(context, "add", " overflow (unsigned)");
    }
  }
  return T{a + b};
}

template <internal::CheckedInteger T>
absl::StatusOr<T> CheckedSub(T a, T b, absl::string_view context = {}) {
  if constexpr (std::is_signed_v<T>) {
    if ((b > 0 && a < std::numeric_limits<T>::min() + b) ||
        (b < 0 && a > std::numeric_limits<T>::max() + b)) {
      return internal::OverflowError(context, "sub", " overflow (signed)");
    }
  } else {
    if (a < b) {
      return internal::OverflowError(context, "sub", " underflow (unsigned)");
    }
  }
  return T{a - b};
}

template <internal::CheckedInteger T>
absl::StatusOr<T> CheckedMul(T a, T b, absl::string_view context = {}) {
  constexpr T kMin = std::numeric_limits<T>::min();
  constexpr T kMax = std::numeric_limits<T>::max();
  if (a == 0 || b == 0) return T{0};
  if constexpr (std::is_signed_v<T>) {
    // Division-based pre-checks only: the product is never computed until it
    // is proven representable (computing it first would be signed overflow).
    bool overflow = false;
    if (b > 0) {
      overflow = a > kMax / b || a < kMin / b;
    } else if (a > 0) {
      overflow = b < kMin / a;
    } else {
      overflow = a < kMax / b;
    }
    if (overflow) {
      return internal::OverflowError(context, "mul", " overflow (signed)");
    }
    return T{a * b};
  } else {
    if (a > kMax / b) {
      return internal::OverflowError(context, "mul", " overflow (unsigned)");
    }
    return T{a * b};
  }
}

// CheckedCeilDiv: b == 0 is InvalidArgument (caller data error, not
// overflow); the result is checked to fit T.
template <internal::CheckedInteger T>
absl::StatusOr<T> CheckedCeilDiv(T a, T b, absl::string_view context = {}) {
  if (b == 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(context.empty() ? "checked_math" : context, ": ceil-div by zero"));
  }
  if constexpr (std::is_signed_v<T>) {
    if (a == std::numeric_limits<T>::min() && b == -1) {
      return internal::OverflowError(context, "ceil-div", " overflow (signed)");
    }
  }
  const T quotient = a / b;
  const T remainder = a % b;
  const bool has_remainder = remainder != 0 && ((remainder > 0) == (b > 0));
  if (!has_remainder) return quotient;
  // quotient + 1 cannot overflow when a remainder exists and signs agree:
  // |result| <= |a| except for the min/-1 case rejected above.
  return T{quotient + 1};
}

// CheckedNarrow<To>: explicit, checked conversion between integer types.
// Bounds are compared in a 64-bit common domain so widening never wraps the
// destination limit into the source representation.
template <internal::CheckedInteger To, internal::CheckedInteger From>
absl::StatusOr<To> CheckedNarrow(From value, absl::string_view context = {}) {
  const std::string detail = absl::StrCat(" value ", value, " outside destination range");
  bool out_of_range = false;  // detail is used only on the failure path
  if constexpr (std::is_signed_v<From>) {
    const long long wide = static_cast<long long>(value);
    if constexpr (std::is_signed_v<To>) {
      out_of_range = wide < static_cast<long long>(std::numeric_limits<To>::min()) ||
                     wide > static_cast<long long>(std::numeric_limits<To>::max());
    } else {
      out_of_range =
          wide < 0 || static_cast<unsigned long long>(wide) >
                          static_cast<unsigned long long>(std::numeric_limits<To>::max());
    }
  } else {
    const unsigned long long wide = static_cast<unsigned long long>(value);
    out_of_range = wide > static_cast<unsigned long long>(std::numeric_limits<To>::max());
  }
  if (out_of_range) {
    return internal::OverflowError(context, "narrow", detail);
  }
  return static_cast<To>(value);
}

// Byte-size arithmetic shared with shape/tensor code: every byte product
// in the codebase routes through here.
inline absl::StatusOr<ByteCount> CheckedByteSize(uint64_t element_count, uint64_t element_width,
                                                 absl::string_view context = {}) {
  absl::StatusOr<uint64_t> bytes = CheckedMul(element_count, element_width, context);
  if (!bytes.ok()) {
    return bytes.status();
  }
  return ByteCount(bytes.value());
}

}  // namespace inferx

#endif  // INFERX_BASE_CHECKED_MATH_H_
