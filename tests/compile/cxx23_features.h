// C++23 feature sentinel declarations (m0.md section 7.2): exercises library
// and language facilities the roadmap relies on, not just compiler macros.
// The missing-facility fixture compiles this header at C++17 to prove the
// diagnostics name the missing facility.
#ifndef INFERX_TESTS_COMPILE_CXX23_FEATURES_H_
#define INFERX_TESTS_COMPILE_CXX23_FEATURES_H_

#include <version>

// --- Feature-test macros with named-facility diagnostics --------------------
// Checked before any facility header is included so a missing library feature
// produces this guidance rather than an opaque header error. Undefined macros
// need #error (static_assert on an undeclared identifier is a hard parse
// error without the message); defined-but-old macros hit the static_asserts.
#define INFERX_REQUIRE_FEATURE(feature_macro, facility)                                    \
  static_assert(feature_macro,                                                             \
                "InferX requires the C++23 " facility                                      \
                " support in the supported standard library (feature test " #feature_macro \
                "). Raise the toolchain floor per ADR 0005 / "                             \
                "docs/supported-platforms.md; do not silently compile "                    \
                "this lane as C++20.")

#if !defined(__cpp_lib_optional)
#error \
    "std::optional monadic operations: InferX requires C++23 std::optional monadic support in the supported standard library (__cpp_lib_optional is undefined). Raise the toolchain floor per ADR 0005 / docs/supported-platforms.md"
#endif
#if !defined(__cpp_lib_unreachable)
#error \
    "std::unreachable: InferX requires the C++23 std::unreachable support in the supported standard library (__cpp_lib_unreachable is undefined). Raise the toolchain floor per ADR 0005 / docs/supported-platforms.md"
#endif
#if !defined(__cpp_lib_string_contains)
#error \
    "std::string::contains: InferX requires the C++23 std::string/std::string_view contains support in the supported standard library (__cpp_lib_string_contains is undefined). Raise the toolchain floor per ADR 0005 / docs/supported-platforms.md"
#endif
#if !defined(__cpp_lib_span)
#error \
    "std::span: InferX requires the C++23 std::span support in the supported standard library (__cpp_lib_span is undefined). Raise the toolchain floor per ADR 0005 / docs/supported-platforms.md"
#endif
#if !defined(__cpp_lib_jthread)
#error \
    "std::jthread: InferX requires the C++23 std::jthread and std::stop_token support in the supported standard library (__cpp_lib_jthread is undefined). Raise the toolchain floor per ADR 0005 / docs/supported-platforms.md"
#endif
#if !defined(__cpp_concepts)
#error \
    "concepts: InferX requires the C++23 concepts support in the compiler (__cpp_concepts is undefined). Raise the toolchain floor per ADR 0005 / docs/supported-platforms.md"
#endif

INFERX_REQUIRE_FEATURE(__cpp_lib_optional, "std::optional monadic operations");
INFERX_REQUIRE_FEATURE(__cpp_lib_unreachable, "std::unreachable");
INFERX_REQUIRE_FEATURE(__cpp_lib_string_contains, "std::string::contains");
INFERX_REQUIRE_FEATURE(__cpp_lib_span, "std::span");
INFERX_REQUIRE_FEATURE(__cpp_lib_jthread, "std::jthread and std::stop_token");
INFERX_REQUIRE_FEATURE(__cpp_concepts, "concepts");

// std::expected: exercised on every lane that exposes it. The Clang 18 +
// libstdc++ 13 pairing cannot: libstdc++ gates <expected> on
// __cpp_concepts >= 202002L while Clang reports 201907L. This is a documented
// facility gap (ADR 0005, docs/supported-platforms.md), not a silent C++20
// fallback; the accepted-alternative C++23 facilities above cover every lane.
#if defined(__cpp_lib_expected)
#define INFERX_HAVE_STD_EXPECTED 1
#else
#define INFERX_HAVE_STD_EXPECTED 0
#endif

#include <concepts>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#if INFERX_HAVE_STD_EXPECTED
#include <expected>
#endif

namespace inferx::testing {

// --- C++23 library facilities available on every accepted lane -------------
inline std::optional<long> OptionalChain(int value) {
  // Monadic optional (C++23): and_then/transform/or_else.
  return std::make_optional(value)
      .and_then(
          [](int v) -> std::optional<int> { return v > 0 ? std::optional<int>(v) : std::nullopt; })
      .transform([](int v) { return static_cast<long>(v) * 10L; })
      .or_else([]() -> std::optional<long> { return std::optional<long>(-1L); });
}

inline int UnreachableGuard(bool reachable) {
  if (!reachable) {
    std::unreachable();  // C++23 <utility>
  }
  return 1;
}

inline bool ContainsToken(std::string_view text, std::string_view token) {
  return text.contains(token);  // C++23 string/string_view
}

enum class ProbeKind : unsigned char { kAlpha = 3, kBeta = 9 };

constexpr int Underlying(ProbeKind kind) { return static_cast<int>(std::to_underlying(kind)); }

// --- std::expected (lanes that expose it; see note above) ------------------
#if INFERX_HAVE_STD_EXPECTED
inline std::expected<int, std::string> ExpectedValue(int value) { return value; }
inline std::expected<int, std::string> ExpectedError(std::string message) {
  return std::unexpected(std::move(message));
}
#endif

// --- Non-owning views: std::span and std::string_view ----------------------
inline long SumSpan(std::span<const int> values) {
  long total = 0;
  for (const int value : values) total += value;
  return total;
}
inline bool HasPrefix(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

// --- Concepts constraint used by later milestones ---------------------------
template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <Numeric T>
constexpr T Doubled(T value) {
  return value * T{2};
}

// --- std::jthread / std::stop_token (cooperative cancellation) --------------
struct StopTokenProbe {
  bool requested = false;
};

inline StopTokenProbe RunUntilStopped() {
  StopTokenProbe probe;
  std::jthread worker([&probe](const std::stop_token& token) {
    while (!token.stop_requested()) {
      std::this_thread::yield();
    }
    probe.requested = true;
  });
  worker.request_stop();
  worker.join();  // read the flag only after the worker observed the stop
  return probe;
}

}  // namespace inferx::testing

#endif  // INFERX_TESTS_COMPILE_CXX23_FEATURES_H_
