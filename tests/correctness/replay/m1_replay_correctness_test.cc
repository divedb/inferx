#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "inferx/api/response_event.h"
#include "inferx/base/token.h"
#include "inferx/config/config_loader.h"
#include "inferx/config/engine_config.h"
#include "inferx/simulator/engine_simulator.h"
#include "inferx/simulator/replay.h"
#include "inferx/simulator/workload.h"

namespace inferx::simulator {
namespace {

const config::ModelCapabilities kModel{TokenCount(32768), false};

struct RunOutput {
  SimulationResult result;
  std::vector<ResponseEvent> responses;
  std::string trace;
};

absl::StatusOr<config::EngineConfig> ReadConfig(const std::string& filename) {
  auto fields = config::ReadConfigFile(std::string(INFERX_M1_REPLAY_DATA_DIR) + "/" + filename);
  if (!fields.ok()) return fields.status();
  auto parsed = config::LoadConfig(*fields, std::nullopt, std::nullopt);
  if (!parsed.ok()) return parsed.status();
  return config::EngineConfig::Validate(*parsed, {}, kModel);
}

absl::StatusOr<RunOutput> RunSimulation(const config::EngineConfig& config,
                                        const std::vector<WorkloadEvent>& workload) {
  InMemoryReplaySink sink(ReplayBufferLimits{.max_bytes = 16U << 20U, .max_records = 8192});
  auto simulator = EngineSimulator::Create(config, kModel, sink);
  if (!simulator.ok()) return simulator.status();
  absl::Status loaded = (*simulator)->LoadWorkload(workload);
  if (!loaded.ok()) return loaded;
  auto result = (*simulator)->Run();
  if (!result.ok()) return result.status();
  return RunOutput{*result,
                   std::vector<ResponseEvent>((*simulator)->responses().begin(),
                                              (*simulator)->responses().end()),
                   sink.contents()};
}

std::vector<std::tuple<uint64_t, uint32_t, int32_t>> Tokens(
    const std::vector<ResponseEvent>& responses) {
  std::vector<std::tuple<uint64_t, uint32_t, int32_t>> tokens;
  for (const ResponseEvent& response : responses) {
    if (const auto* token = std::get_if<TokenDelta>(&response)) {
      tokens.emplace_back(token->request.value(), token->output_position.value(),
                          token->token.value());
    }
  }
  std::sort(tokens.begin(), tokens.end());
  return tokens;
}

TEST(ReplayCorrectnessTest, ReviewedGoldenTraceMatchesSerializerExactly) {
  auto config = ReadConfig("basic_config.json");
  ASSERT_TRUE(config.ok()) << config.status();
  auto workload =
      ReadWorkloadFile(std::string(INFERX_M1_REPLAY_DATA_DIR) + "/basic_workload.jsonl");
  ASSERT_TRUE(workload.ok()) << workload.status();
  auto golden = ReadReplayInputs(std::string(INFERX_M1_REPLAY_DATA_DIR) + "/basic_trace.jsonl");
  ASSERT_TRUE(golden.ok()) << golden.status();
  auto output = RunSimulation(*config, *workload);
  ASSERT_TRUE(output.ok()) << output.status();
  EXPECT_TRUE(output->result.status.ok()) << output->result.status;
  EXPECT_EQ(output->trace, golden->original_bytes);
}

TEST(ReplayCorrectnessTest, ReorderedConfigAndRepeatedRunsAreByteIdentical) {
  auto first_config = ReadConfig("basic_config.json");
  auto second_config = ReadConfig("basic_config_reordered.json");
  ASSERT_TRUE(first_config.ok()) << first_config.status();
  ASSERT_TRUE(second_config.ok()) << second_config.status();
  auto workload =
      ReadWorkloadFile(std::string(INFERX_M1_REPLAY_DATA_DIR) + "/basic_workload.jsonl");
  ASSERT_TRUE(workload.ok()) << workload.status();
  auto first = RunSimulation(*first_config, *workload);
  auto second = RunSimulation(*second_config, *workload);
  auto third = RunSimulation(*first_config, *workload);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(third.ok()) << third.status();
  EXPECT_EQ(first->trace, second->trace);
  EXPECT_EQ(first->trace, third->trace);
}

TEST(ReplayCorrectnessTest, BatchedAndSerialFakeOutputsAreTokenIdentical) {
  auto workload = ParseWorkloadJsonLines(
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":10,\"model_id\":0,\"token_ids\":[1,2],\"max_output_tokens\":3}\n"
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":20,\"model_id\":0,\"token_ids\":[3,4],\"max_output_tokens\":3}\n");
  ASSERT_TRUE(workload.ok()) << workload.status();
  config::ParsedConfig batched_values;
  batched_values.SetMaxQueuedRequests(8, config::ConfigSource::kCommandLine);
  batched_values.SetMaxActiveSequences(2, config::ConfigSource::kCommandLine);
  batched_values.SetMaxSequencesPerStep(2, config::ConfigSource::kCommandLine);
  config::ParsedConfig serial_values = batched_values;
  serial_values.SetMaxSequencesPerStep(1, config::ConfigSource::kCommandLine);
  auto batched_config = config::EngineConfig::Validate(batched_values, {}, kModel);
  auto serial_config = config::EngineConfig::Validate(serial_values, {}, kModel);
  ASSERT_TRUE(batched_config.ok()) << batched_config.status();
  ASSERT_TRUE(serial_config.ok()) << serial_config.status();
  auto batched = RunSimulation(*batched_config, *workload);
  auto serial = RunSimulation(*serial_config, *workload);
  ASSERT_TRUE(batched.ok()) << batched.status();
  ASSERT_TRUE(serial.ok()) << serial.status();
  EXPECT_EQ(Tokens(batched->responses), Tokens(serial->responses));
}

}  // namespace
}  // namespace inferx::simulator
