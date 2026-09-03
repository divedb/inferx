#include <type_traits>
#include <variant>

#include "common.h"

namespace inferx::command {

// One implementation file per command under src/command/.
void Serve(const ServeOptions&);
void Benchmark(const BenchmarkOptions&);
void Run(const GlobalOptions& global, const RunOptions& options);
void Chat(const ChatOptions&);
void Version();
void CollectEnv();

void Dispatch(const Invocation& invocation) {
  std::visit(
      [&invocation](const auto& options) {
        using Selected = std::decay_t<decltype(options)>;
        if constexpr (std::is_same_v<Selected, ServeOptions>) {
          Serve(options);
        } else if constexpr (std::is_same_v<Selected, BenchmarkOptions>) {
          Benchmark(options);
        } else if constexpr (std::is_same_v<Selected, RunOptions>) {
          Run(invocation.global, options);
        } else if constexpr (std::is_same_v<Selected, ChatOptions>) {
          Chat(options);
        } else if constexpr (std::is_same_v<Selected, VersionOptions>) {
          Version();
        } else if constexpr (std::is_same_v<Selected, CollectEnvOptions>) {
          CollectEnv();
        } else {
          static_assert(always_false<Selected>::value, "non-exhaustive visitor!");
        }
      },
      invocation.options);
}

}  // namespace inferx::command
