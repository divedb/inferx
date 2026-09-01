#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/api/response_event.h"
#include "inferx/base/token.h"
#include "inferx/config/engine_config.h"
#include "inferx/config/parsed_config.h"
#include "inferx/simulator/engine_simulator.h"
#include "inferx/simulator/replay.h"
#include "inferx/simulator/workload.h"

namespace inferx::simulator {
namespace {

struct TestConfigValues {
  uint64_t active_sequences = 64;
  uint64_t kv_tokens = 1114112;
  uint64_t sequences_per_step = 32;
};

config::EngineConfig TestConfig(TestConfigValues values = {}) {
  config::ParsedConfig parsed;
  parsed.SetMaxActiveSequences(values.active_sequences, config::ConfigSource::kCommandLine);
  parsed.SetMaxQueuedRequests(64, config::ConfigSource::kCommandLine);
  parsed.SetMaxSequencesPerStep(values.sequences_per_step, config::ConfigSource::kCommandLine);
  parsed.SetSimulatedKvTokenCapacity(values.kv_tokens, config::ConfigSource::kCommandLine);
  parsed.SetMaxSimulationEvents(10000, config::ConfigSource::kCommandLine);
  auto config = config::EngineConfig::Validate(parsed, config::BuildCapabilities{},
                                               config::ModelCapabilities{TokenCount(32768), false});
  EXPECT_TRUE(config.ok()) << config.status();
  return *config;
}

struct ScenarioOutcome {
  absl::Status harness_status;
  SimulationResult simulation;
  std::vector<ResponseEvent> responses;
  std::string replay;
};

ScenarioOutcome RunScenario(const std::string& workload_text,
                            config::EngineConfig config = TestConfig()) {
  auto workload = ParseWorkloadJsonLines(workload_text);
  if (!workload.ok()) {
    return {workload.status(), {workload.status(), {}}, {}, {}};
  }
  InMemoryReplaySink replay(ReplayBufferLimits{.max_bytes = 8U << 20U, .max_records = 4096});
  auto simulator =
      EngineSimulator::Create(config, config::ModelCapabilities{TokenCount(32768), false}, replay);
  if (!simulator.ok()) {
    return {simulator.status(), {simulator.status(), {}}, {}, {}};
  }
  absl::Status loaded = (*simulator)->LoadWorkload(*workload);
  if (!loaded.ok()) {
    return {loaded, {loaded, {}}, {}, {}};
  }
  auto result = (*simulator)->Run();
  if (!result.ok()) {
    return {result.status(), {result.status(), {}}, {}, replay.contents()};
  }
  std::vector<ResponseEvent> responses((*simulator)->responses().begin(),
                                       (*simulator)->responses().end());
  return {absl::OkStatus(), *result, std::move(responses), replay.contents()};
}

std::vector<TerminalResponse> Terminals(const std::vector<ResponseEvent>& responses) {
  std::vector<TerminalResponse> terminals;
  for (const ResponseEvent& response : responses) {
    if (const auto* terminal = std::get_if<TerminalResponse>(&response)) {
      terminals.push_back(*terminal);
    }
  }
  return terminals;
}

void ExpectClean(const ScenarioOutcome& outcome) {
  EXPECT_EQ(outcome.simulation.summary.used_sequences, 0U);
  EXPECT_EQ(outcome.simulation.summary.used_kv_tokens, 0U);
  EXPECT_EQ(outcome.simulation.summary.live_tickets, 0U);
  EXPECT_EQ(outcome.simulation.summary.leased_plan_slots, 0U);
  EXPECT_TRUE(outcome.simulation.summary.invariants_ok);
  EXPECT_TRUE(CheckReplayJsonLines(outcome.replay).ok());
}

TEST(EngineSimulatorIntegrationTest, OneRequestPrefillDecodeAndLengthTerminal) {
  auto workload = ParseWorkloadJsonLines(
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":1,\"model_id\":0,\"token_ids\":[10,11,12],"
      "\"max_output_tokens\":2,\"deadline_ns\":null,"
      "\"finish_after_tokens\":null,\"priority\":0,\"tenant_scope\":0}\n");
  ASSERT_TRUE(workload.ok()) << workload.status();
  InMemoryReplaySink replay(ReplayBufferLimits{.max_bytes = 1U << 20U, .max_records = 256});
  auto simulator = EngineSimulator::Create(
      TestConfig(), config::ModelCapabilities{TokenCount(32768), false}, replay);
  ASSERT_TRUE(simulator.ok()) << simulator.status();
  ASSERT_TRUE((*simulator)->LoadWorkload(*workload).ok());
  auto result = (*simulator)->Run();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->status.ok()) << result->status;
  EXPECT_EQ(result->summary.requests, 1U);
  EXPECT_EQ(result->summary.finished, 1U);
  EXPECT_EQ(result->summary.cancelled, 0U);
  EXPECT_EQ(result->summary.failed, 0U);
  EXPECT_EQ(result->summary.used_sequences, 0U);
  EXPECT_EQ(result->summary.used_kv_tokens, 0U);
  EXPECT_EQ(result->summary.live_tickets, 0U);
  EXPECT_EQ(result->summary.leased_plan_slots, 0U);
  EXPECT_TRUE(result->summary.invariants_ok);

  const auto responses = (*simulator)->responses();
  ASSERT_EQ(responses.size(), 3U);
  EXPECT_TRUE(std::holds_alternative<TokenDelta>(responses[0]));
  EXPECT_TRUE(std::holds_alternative<TokenDelta>(responses[1]));
  ASSERT_TRUE(std::holds_alternative<TerminalResponse>(responses[2]));
  EXPECT_EQ(std::get<TerminalResponse>(responses[2]).reason, FinishReason::kLength);
  EXPECT_TRUE(CheckReplayJsonLines(replay.contents()).ok());
}

TEST(EngineSimulatorIntegrationTest, SameTimeRequestsRemainFcfsUnderConstrainedBatching) {
  const ScenarioOutcome outcome = RunScenario(
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":1,\"model_id\":0,\"token_ids\":[1,2],"
      "\"max_output_tokens\":1}\n"
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":2,\"model_id\":0,\"token_ids\":[3,4],"
      "\"max_output_tokens\":1}\n",
      TestConfig(
          TestConfigValues{.active_sequences = 2, .kv_tokens = 16, .sequences_per_step = 1}));
  ASSERT_TRUE(outcome.harness_status.ok()) << outcome.harness_status;
  EXPECT_TRUE(outcome.simulation.status.ok()) << outcome.simulation.status;
  EXPECT_EQ(outcome.simulation.summary.finished, 2U);
  const auto terminals = Terminals(outcome.responses);
  ASSERT_EQ(terminals.size(), 2U);
  EXPECT_EQ(terminals[0].request, RequestId(1));
  EXPECT_EQ(terminals[1].request, RequestId(2));
  const size_t first_plan = outcome.replay.find("\"record_type\":\"step_plan\"");
  ASSERT_NE(first_plan, std::string::npos);
  EXPECT_NE(outcome.replay.find("\"request_id\":1", first_plan), std::string::npos);
  ExpectClean(outcome);
}

TEST(EngineSimulatorIntegrationTest, CapacityBlockClearsAfterEarlierTerminalRelease) {
  const ScenarioOutcome outcome = RunScenario(
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":1,\"model_id\":0,\"token_ids\":[1,2],"
      "\"max_output_tokens\":1}\n"
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":2,\"model_id\":0,\"token_ids\":[3,4],"
      "\"max_output_tokens\":1}\n",
      TestConfig(TestConfigValues{.active_sequences = 1, .kv_tokens = 3, .sequences_per_step = 1}));
  ASSERT_TRUE(outcome.harness_status.ok()) << outcome.harness_status;
  EXPECT_TRUE(outcome.simulation.status.ok()) << outcome.simulation.status;
  EXPECT_EQ(outcome.simulation.summary.finished, 2U);
  const size_t released_first =
      outcome.replay.find("\"action\":\"release\",\"reservation_id\":1,\"request_id\":1");
  const size_t reserved_second =
      outcome.replay.find("\"action\":\"reserve\",\"reservation_id\":2,\"request_id\":2");
  ASSERT_NE(released_first, std::string::npos);
  ASSERT_NE(reserved_second, std::string::npos);
  EXPECT_LT(released_first, reserved_second);
  ExpectClean(outcome);
}

TEST(EngineSimulatorIntegrationTest, CancelsQueuedPrefillingAndDecodingWithoutLeaks) {
  const ScenarioOutcome queued = RunScenario(
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":1,\"model_id\":0,\"token_ids\":[1,2],\"max_output_tokens\":1}\n"
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":2,\"model_id\":0,\"token_ids\":[3,4],\"max_output_tokens\":1}\n"
      "{\"schema_version\":1,\"event_type\":\"cancel\",\"at_ns\":10,"
      "\"request_id\":2}\n",
      TestConfig(TestConfigValues{.active_sequences = 1, .kv_tokens = 3, .sequences_per_step = 1}));
  ASSERT_TRUE(queued.harness_status.ok()) << queued.harness_status;
  EXPECT_TRUE(queued.simulation.status.ok()) << queued.simulation.status;
  EXPECT_EQ(queued.simulation.summary.finished, 1U);
  EXPECT_EQ(queued.simulation.summary.cancelled, 1U);
  ExpectClean(queued);

  const ScenarioOutcome prefill = RunScenario(
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":3,\"model_id\":0,\"token_ids\":[1,2],\"max_output_tokens\":2}\n"
      "{\"schema_version\":1,\"event_type\":\"cancel\",\"at_ns\":10,"
      "\"request_id\":3}\n");
  ASSERT_TRUE(prefill.harness_status.ok()) << prefill.harness_status;
  EXPECT_TRUE(prefill.simulation.status.ok()) << prefill.simulation.status;
  ASSERT_EQ(Terminals(prefill.responses).size(), 1U);
  EXPECT_EQ(Terminals(prefill.responses)[0].reason, FinishReason::kCancelled);
  EXPECT_EQ(prefill.responses.size(), 1U);
  ExpectClean(prefill);

  const ScenarioOutcome decode = RunScenario(
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":4,\"model_id\":0,\"token_ids\":[1,2],\"max_output_tokens\":2}\n"
      "{\"schema_version\":1,\"event_type\":\"cancel\",\"at_ns\":1250,"
      "\"request_id\":4}\n");
  ASSERT_TRUE(decode.harness_status.ok()) << decode.harness_status;
  EXPECT_TRUE(decode.simulation.status.ok()) << decode.simulation.status;
  ASSERT_EQ(Terminals(decode.responses).size(), 1U);
  EXPECT_EQ(Terminals(decode.responses)[0].reason, FinishReason::kCancelled);
  EXPECT_EQ(decode.responses.size(), 2U);
  ExpectClean(decode);
}

TEST(EngineSimulatorIntegrationTest, ExactDeadlineWinsOverCompletion) {
  const ScenarioOutcome outcome = RunScenario(
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":1,\"model_id\":0,\"token_ids\":[1,2],\"max_output_tokens\":1,"
      "\"deadline_ns\":1200}\n");
  ASSERT_TRUE(outcome.harness_status.ok()) << outcome.harness_status;
  EXPECT_TRUE(outcome.simulation.status.ok()) << outcome.simulation.status;
  const auto terminals = Terminals(outcome.responses);
  ASSERT_EQ(terminals.size(), 1U);
  EXPECT_EQ(terminals[0].reason, FinishReason::kDeadline);
  EXPECT_EQ(terminals[0].terminal_time, MonotonicTime(Nanoseconds(1200)));
  EXPECT_EQ(outcome.responses.size(), 1U);
  ExpectClean(outcome);
}

TEST(EngineSimulatorIntegrationTest, PreemptRequeueIncrementsEpochAndCompletes) {
  const ScenarioOutcome outcome = RunScenario(
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":1,\"model_id\":0,\"token_ids\":[1,2],\"max_output_tokens\":2}\n"
      "{\"schema_version\":1,\"event_type\":\"preempt\",\"at_ns\":1200,"
      "\"request_id\":1}\n"
      "{\"schema_version\":1,\"event_type\":\"requeue\",\"at_ns\":1200,"
      "\"request_id\":1}\n");
  ASSERT_TRUE(outcome.harness_status.ok()) << outcome.harness_status;
  EXPECT_TRUE(outcome.simulation.status.ok()) << outcome.simulation.status;
  EXPECT_EQ(outcome.simulation.summary.finished, 1U);
  EXPECT_NE(outcome.replay.find("\"event\":\"requeue\",\"to\":\"queued\""), std::string::npos);
  EXPECT_NE(outcome.replay.find("\"epoch\":1"), std::string::npos);
  ExpectClean(outcome);
}

TEST(EngineSimulatorIntegrationTest, ExecutorFailureIsReportedAfterCleanTeardown) {
  const ScenarioOutcome outcome = RunScenario(
      "{\"schema_version\":1,\"event_type\":\"inject_executor_failure\",\"at_ns\":0,"
      "\"target_kind\":\"request\",\"target_id\":1,\"status_code\":13,"
      "\"error_reason\":\"executor-failure\",\"persistent\":false}\n"
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":1,\"model_id\":0,\"token_ids\":[1,2],\"max_output_tokens\":2}\n");
  ASSERT_TRUE(outcome.harness_status.ok()) << outcome.harness_status;
  EXPECT_FALSE(outcome.simulation.status.ok());
  EXPECT_EQ(GetErrorReason(outcome.simulation.status).value(), ErrorReason::kExecutorFailure);
  EXPECT_EQ(outcome.simulation.summary.failed, 1U);
  ASSERT_EQ(Terminals(outcome.responses).size(), 1U);
  EXPECT_EQ(Terminals(outcome.responses)[0].reason, FinishReason::kExecutorError);
  ExpectClean(outcome);
}

TEST(EngineSimulatorIntegrationTest, ShutdownCancelAndDrainHaveDistinctResults) {
  const std::string submit =
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":1,\"model_id\":0,\"token_ids\":[1,2],\"max_output_tokens\":2}\n";
  const ScenarioOutcome cancelled =
      RunScenario(submit +
                  "{\"schema_version\":1,\"event_type\":\"shutdown\",\"at_ns\":10,"
                  "\"mode\":\"cancel\"}\n");
  ASSERT_TRUE(cancelled.harness_status.ok()) << cancelled.harness_status;
  EXPECT_TRUE(cancelled.simulation.status.ok()) << cancelled.simulation.status;
  EXPECT_EQ(cancelled.simulation.summary.cancelled, 1U);
  ASSERT_EQ(Terminals(cancelled.responses).size(), 1U);
  EXPECT_EQ(Terminals(cancelled.responses)[0].reason, FinishReason::kShutdown);
  ExpectClean(cancelled);

  const ScenarioOutcome drained =
      RunScenario(submit +
                  "{\"schema_version\":1,\"event_type\":\"shutdown\",\"at_ns\":10,"
                  "\"mode\":\"drain\"}\n");
  ASSERT_TRUE(drained.harness_status.ok()) << drained.harness_status;
  EXPECT_TRUE(drained.simulation.status.ok()) << drained.simulation.status;
  EXPECT_EQ(drained.simulation.summary.finished, 1U);
  ASSERT_EQ(Terminals(drained.responses).size(), 1U);
  EXPECT_EQ(Terminals(drained.responses)[0].reason, FinishReason::kLength);
  ExpectClean(drained);
}

TEST(EngineSimulatorIntegrationTest, IdenticalInputsProduceByteIdenticalReplay) {
  const std::string workload =
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":9,\"model_id\":0,\"token_ids\":[1,2,3],\"max_output_tokens\":2,"
      "\"finish_after_tokens\":1}\n";
  const ScenarioOutcome first = RunScenario(workload);
  const ScenarioOutcome second = RunScenario(workload);
  ASSERT_TRUE(first.harness_status.ok()) << first.harness_status;
  ASSERT_TRUE(second.harness_status.ok()) << second.harness_status;
  EXPECT_TRUE(first.simulation.status.ok()) << first.simulation.status;
  EXPECT_TRUE(second.simulation.status.ok()) << second.simulation.status;
  EXPECT_EQ(first.replay, second.replay);
  ASSERT_EQ(Terminals(first.responses).size(), 1U);
  EXPECT_EQ(Terminals(first.responses)[0].reason, FinishReason::kSimulatedEos);
  ExpectClean(first);
  ExpectClean(second);
}

}  // namespace
}  // namespace inferx::simulator
