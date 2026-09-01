#include <benchmark/benchmark.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <span>
#include <vector>

#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/scheduler/batch_planner.h"
#include "inferx/scheduler/fcfs_policy.h"
#include "inferx/scheduler/resource_accountant.h"
#include "inferx/scheduler/scheduling_snapshot.h"
#include "inferx/scheduler/step_plan.h"

namespace {

std::atomic<uint64_t> g_allocations{0};
thread_local bool g_measure_allocations = false;

void CountAllocation() noexcept {
  if (g_measure_allocations) g_allocations.fetch_add(1, std::memory_order_relaxed);
}

enum class Scenario : uint8_t { kPrefill, kDecode, kMixed, kSaturated };

double Percentile(const std::vector<double>& samples, double fraction) {
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const size_t index = static_cast<size_t>(fraction * static_cast<double>(sorted.size() - 1));
  return sorted[index];
}

double P50(const std::vector<double>& samples) { return Percentile(samples, 0.50); }
double P90(const std::vector<double>& samples) { return Percentile(samples, 0.90); }
double P99(const std::vector<double>& samples) { return Percentile(samples, 0.99); }

struct Fixture {
  std::vector<inferx::scheduler::RequestSchedulingView> views;
  std::vector<size_t> selection;
  std::unique_ptr<inferx::scheduler::StepPlanPool> pool;
  inferx::scheduler::SchedulingLimits limits{inferx::SequenceCount(0), inferx::TokenCount(0)};
};

Fixture MakeFixture(size_t count, Scenario scenario) {
  Fixture fixture;
  fixture.views.reserve(count);
  fixture.selection.resize(count);
  const uint32_t sequence_limit = scenario == Scenario::kSaturated
                                      ? std::max<uint32_t>(1, static_cast<uint32_t>(count / 2))
                                      : static_cast<uint32_t>(count);
  fixture.limits = {inferx::SequenceCount(sequence_limit), inferx::TokenCount(4096)};
  auto pool = inferx::scheduler::StepPlanPool::Create(
      1, inferx::SequenceCount(static_cast<uint32_t>(count)));
  if (pool.ok()) fixture.pool = std::move(*pool);
  for (size_t index = 0; index < count; ++index) {
    inferx::RequestState state = inferx::RequestState::kPrefillReady;
    uint32_t committed = 0;
    if (scenario == Scenario::kDecode) {
      state = inferx::RequestState::kDecodeReady;
      committed = 1;
    } else if (scenario == Scenario::kMixed) {
      if (index % 3 == 0) {
        state = inferx::RequestState::kReceived;
      } else if (index % 3 == 1) {
        state = inferx::RequestState::kDecodeReady;
        committed = 1;
      }
    }
    fixture.views.push_back(inferx::scheduler::RequestSchedulingView{
        .request = inferx::RequestId(index + 1),
        .sequence = inferx::SequenceId(index + 1),
        .epoch = inferx::RequestEpoch(0),
        .model = inferx::ModelId(0),
        .state = state,
        .arrival_ordinal = index,
        .arrival_time = inferx::MonotonicTime(inferx::Nanoseconds(0)),
        .prompt_tokens = inferx::TokenCount(4),
        .computed_tokens = inferx::TokenCount(state == inferx::RequestState::kDecodeReady ? 4 : 0),
        .committed_output_tokens = inferx::TokenCount(committed),
        .max_output_tokens = inferx::TokenCount(8),
        .deadline = std::nullopt,
        .reservation = state == inferx::RequestState::kReceived
                           ? std::nullopt
                           : std::optional<inferx::ReservationId>(inferx::ReservationId(index + 1)),
    });
  }
  return fixture;
}

void PlanningBenchmark(benchmark::State& state, Scenario scenario) {
  const size_t count = static_cast<size_t>(state.range(0));
  Fixture fixture = MakeFixture(count, scenario);
  if (fixture.pool == nullptr) {
    state.SkipWithError("cannot create plan pool");
    return;
  }
  inferx::scheduler::FcfsPolicy policy;
  inferx::scheduler::BatchPlanner planner;
  const inferx::scheduler::SchedulingSnapshot snapshot{
      .now = inferx::MonotonicTime(inferx::Nanoseconds(0)),
      .proposed_step = inferx::StepId(1),
      .requests = fixture.views,
      .limits = fixture.limits,
      .resources =
          inferx::scheduler::ResourceSnapshot{
              inferx::SequenceCount(static_cast<uint32_t>(count)), inferx::SequenceCount(0),
              inferx::KvTokenCount(1'000'000), inferx::KvTokenCount(0)},
  };

  uint64_t allocations = 0;
  uint64_t items_planned = 0;
  for (auto iteration : state) {
    static_cast<void>(iteration);
    g_allocations.store(0, std::memory_order_relaxed);
    g_measure_allocations = true;
    auto selected = policy.Select(snapshot, fixture.selection);
    auto lease = fixture.pool->Acquire(snapshot.proposed_step, inferx::ModelId(0), snapshot.now);
    bool ok = selected.ok() && lease.ok();
    if (ok) {
      const absl::Status planned = planner.Build(
          snapshot, std::span<const size_t>(fixture.selection.data(), *selected), *lease);
      ok = planned.ok();
      if (ok) items_planned += lease->plan().sequences.size();
    }
    g_measure_allocations = false;
    allocations += g_allocations.load(std::memory_order_relaxed);
    if (!ok) {
      state.SkipWithError("scheduler planning failed");
      break;
    }
    benchmark::DoNotOptimize(items_planned);
  }
  state.counters["allocations"] = static_cast<double>(allocations);
  state.counters["requests_inspected"] =
      benchmark::Counter(static_cast<double>(count), benchmark::Counter::kIsIterationInvariantRate);
  state.counters["items_planned"] =
      benchmark::Counter(static_cast<double>(items_planned), benchmark::Counter::kIsRate);
  state.counters["tokens_planned"] =
      benchmark::Counter(static_cast<double>(items_planned), benchmark::Counter::kIsRate);
}

void Prefill(benchmark::State& state) { PlanningBenchmark(state, Scenario::kPrefill); }
void Decode(benchmark::State& state) { PlanningBenchmark(state, Scenario::kDecode); }
void Mixed(benchmark::State& state) { PlanningBenchmark(state, Scenario::kMixed); }
void Saturated(benchmark::State& state) { PlanningBenchmark(state, Scenario::kSaturated); }

benchmark::Benchmark* Configure(benchmark::Benchmark* benchmark) {
  return benchmark->Arg(1)
      ->Arg(32)
      ->Arg(256)
      ->Arg(1024)
      ->Repetitions(30)
      ->ReportAggregatesOnly()
      ->ComputeStatistics("p50", P50)
      ->ComputeStatistics("p90", P90)
      ->ComputeStatistics("p99", P99);
}

}  // namespace

void* operator new(std::size_t size) {
  CountAllocation();
  if (void* memory = std::malloc(size)) return memory;
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
  CountAllocation();
  if (void* memory = std::malloc(size)) return memory;
  throw std::bad_alloc();
}

// NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete) -- intentional matching global delete.
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t size) noexcept {
  static_cast<void>(size);
  std::free(memory);
}
void operator delete[](void* memory, std::size_t size) noexcept {
  static_cast<void>(size);
  std::free(memory);
}

BENCHMARK(Prefill)->Apply(Configure);
BENCHMARK(Decode)->Apply(Configure);
BENCHMARK(Mixed)->Apply(Configure);
BENCHMARK(Saturated)->Apply(Configure);

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::AddCustomContext("inferx_milestone", "M1");
  benchmark::AddCustomContext(
      "m1_config", "schema-v1;max_sequences_per_step=request-count;max_tokens_per_step=4096");
  benchmark::AddCustomContext("m1_workload", "prefill,decode,mixed,saturated;1,32,256,1024");
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
