// C++23 sentinel main (m0.md section 7.2): runs the feature probes so the
// supported standard library is verified at runtime, not just the front end.
#include <cstdio>
#include <optional>
#include <span>

#include "cxx23_features.h"

namespace {

int Fail(const char* facility) {
  std::fprintf(stderr,
               "cxx23 sentinel failed: %s is not usable at runtime; the "
               "standard library does not meet the C++23 floor (ADR 0020)\n",
               facility);
  return 1;
}

}  // namespace

int main() {
  using namespace inferx::testing;

  const std::optional<long> chained = OptionalChain(21);
  const std::optional<long> fallback = OptionalChain(-1);
  if (!chained.has_value() || *chained != 210L || !fallback.has_value() || *fallback != -1L) {
    return Fail("std::optional monadic operations");
  }

  if (UnreachableGuard(true) != 1 || !ContainsToken("inferx:sentinel", "sentinel") ||
      ContainsToken("inferx:sentinel", "absent") || Underlying(ProbeKind::kBeta) != 9) {
    return Fail("std::unreachable / std::string::contains / std::to_underlying");
  }

#if INFERX_HAVE_STD_EXPECTED
  const auto ok_value = ExpectedValue(21);
  const auto ok_error = ExpectedError("boom");
  if (!ok_value.has_value() || ok_value.value() != 21 || ok_error.has_value() ||
      ok_error.error() != "boom") {
    return Fail("std::expected");
  }
#endif

  static constexpr int kValues[] = {1, 2, 3, 4, 5};
  if (SumSpan(std::span<const int>(kValues)) != 15 || !HasPrefix("inferx", "inf") ||
      HasPrefix("inf", "inferx")) {
    return Fail("std::span / std::string_view");
  }

  static_assert(Doubled(21) == 42, "concept-constrained constexpr failed");
  if constexpr (Doubled(2.5) != 5.0) {
    return Fail("concepts");
  }

  if (!RunUntilStopped().requested) {
    return Fail("std::jthread / std::stop_token");
  }

#if INFERX_HAVE_STD_EXPECTED
  std::puts(
      "cxx23 sentinel ok: optional-monadic, unreachable, string-contains, "
      "to_underlying, expected, span, string_view, concepts, jthread");
#else
  std::puts(
      "cxx23 sentinel ok: optional-monadic, unreachable, string-contains, "
      "to_underlying, span, string_view, concepts, jthread "
      "(std::expected unavailable on this lane; see ADR 0020)");
#endif
  return 0;
}
