// Base/startup benchmark smoke (m0.md section 7.3): enough calls to prove
// Google Benchmark registration and machine-readable JSON output. It does not
// claim a meaningful runtime performance number.
#include <benchmark/benchmark.h>

#include "inferx/base/version.h"

namespace {

void BM_GetVersion(benchmark::State& state) {
  // '_' is the idiomatic benchmark loop variable; the analyzer flags its
  // dead store, but the range expression drives the measurement loop.
  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    benchmark::DoNotOptimize(inferx::GetVersion());
  }
}
BENCHMARK(BM_GetVersion);

void BM_GetVersionString(benchmark::State& state) {
  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    benchmark::DoNotOptimize(inferx::GetVersionString());
  }
}
BENCHMARK(BM_GetVersionString);

}  // namespace

BENCHMARK_MAIN();
