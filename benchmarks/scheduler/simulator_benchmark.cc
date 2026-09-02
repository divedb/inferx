#include <benchmark/benchmark.h>
#include <sys/resource.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "inferx/api/generate_request.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/config/engine_config.h"
#include "inferx/config/parsed_config.h"
#include "inferx/simulator/engine_simulator.h"
#include "inferx/simulator/replay.h"
#include "inferx/simulator/workload.h"

namespace {

class DiscardReplaySink final : public inferx::simulator::ReplaySink {
 public:
  absl::Status WriteLine(absl::string_view line) override {
    bytes_ += line.size() + 1;
    return absl::OkStatus();
  }
  absl::Status Close() override { return absl::OkStatus(); }
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

 private:
  uint64_t bytes_ = 0;
};

inferx::config::EngineConfig Config(size_t requests) {
  inferx::config::ParsedConfig parsed;
  const uint64_t capacity = std::max<uint64_t>(requests, 1);
  parsed.SetMaxQueuedRequests(capacity, inferx::config::ConfigSource::kCommandLine);
  parsed.SetMaxActiveSequences(capacity, inferx::config::ConfigSource::kCommandLine);
  parsed.SetMaxSequencesPerStep(std::min<uint64_t>(capacity, 1024),
                                inferx::config::ConfigSource::kCommandLine);
  parsed.SetMaxSimulationEvents(std::max<uint64_t>(capacity * 16, 1000),
                                inferx::config::ConfigSource::kCommandLine);
  parsed.SetSimulatedKvTokenCapacity(capacity * 16, inferx::config::ConfigSource::kCommandLine);
  auto config = inferx::config::EngineConfig::Validate(
      parsed, {}, {.max_context_tokens = inferx::TokenCount(32768), .accepts_text = false});
  return *config;
}

std::vector<inferx::simulator::WorkloadEvent> Workload(size_t requests) {
  std::vector<inferx::simulator::WorkloadEvent> events;
  events.reserve(requests);
  for (size_t index = 0; index < requests; ++index) {
    inferx::GenerateRequest request{
        .id = inferx::RequestId(index + 1),
        .model = inferx::ModelId(0),
        .input = std::vector<inferx::TokenId>{inferx::TokenId(1), inferx::TokenId(2)},
        .generation = {.max_output_tokens = inferx::TokenCount(2)},
        .priority = inferx::Priority(0),
        .deadline = std::nullopt,
        .tenant = inferx::TenantScope(0),
    };
    events.push_back(inferx::simulator::WorkloadEvent{
        1, inferx::simulator::WorkloadEventType::kSubmit,
        inferx::MonotonicTime(inferx::Nanoseconds(0)), index,
        inferx::simulator::SubmitWorkloadEvent{std::move(request), std::nullopt}});
  }
  return events;
}

void RunSimulator(benchmark::State& state) {
  const size_t requests = static_cast<size_t>(state.range(0));
  const bool retain_trace = state.range(1) != 0;
  const auto config = Config(requests);
  const auto workload = Workload(requests);
  uint64_t events = 0;
  uint64_t trace_bytes = 0;
  for (auto iteration : state) {
    static_cast<void>(iteration);
    DiscardReplaySink discard;
    inferx::simulator::InMemoryReplaySink retained(inferx::simulator::ReplayBufferLimits{
        .max_bytes = size_t{64} * 1024U * 1024U,
        .max_records = requests * size_t{32} + size_t{16},
    });
    inferx::simulator::ReplaySink& sink =
        retain_trace ? static_cast<inferx::simulator::ReplaySink&>(retained)
                     : static_cast<inferx::simulator::ReplaySink&>(discard);
    auto simulator = inferx::simulator::EngineSimulator::Create(
        config, {.max_context_tokens = inferx::TokenCount(32768), .accepts_text = false}, sink);
    if (!simulator.ok() || !(*simulator)->LoadWorkload(workload).ok()) {
      state.SkipWithError("simulator setup failed");
      break;
    }
    auto result = (*simulator)->Run();
    if (!result.ok() || !result->status.ok()) {
      state.SkipWithError("simulator run failed");
      break;
    }
    events += result->summary.events;
    trace_bytes += retain_trace ? retained.contents().size() : discard.bytes();
    benchmark::DoNotOptimize(result->summary.finished);
  }
  rusage usage{};
  static_cast<void>(getrusage(RUSAGE_SELF, &usage));
  state.counters["events_per_second"] =
      benchmark::Counter(static_cast<double>(events), benchmark::Counter::kIsRate);
  state.counters["peak_rss_kib"] = static_cast<double>(usage.ru_maxrss);
  state.counters["requests"] = static_cast<double>(requests);
  state.counters["trace_bytes_per_second"] =
      benchmark::Counter(static_cast<double>(trace_bytes), benchmark::Counter::kIsRate);
}

}  // namespace

BENCHMARK(RunSimulator)
    ->Args({1, 0})
    ->Args({1, 1})
    ->Args({32, 0})
    ->Args({32, 1})
    ->Args({256, 0})
    ->Args({256, 1});

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::AddCustomContext("inferx_config", "schema-v1;fake-model=0;context=32768");
  benchmark::AddCustomContext("inferx_workload", "prompt=2;output=2;requests=1,32,256");
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
