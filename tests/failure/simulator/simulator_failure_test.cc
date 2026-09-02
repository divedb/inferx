#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "absl/status/status.h"
#include "inferx/base/token.h"
#include "inferx/config/config_loader.h"
#include "inferx/config/engine_config.h"
#include "inferx/config/parsed_config.h"
#include "inferx/simulator/engine_simulator.h"
#include "inferx/simulator/replay.h"
#include "inferx/simulator/workload.h"

namespace inferx::simulator {
namespace {

TEST(SimulatorFailureTest, InvalidWorkloadFormsFailDeterministically) {
  EXPECT_FALSE(ParseWorkloadJsonLines("\n").ok());
  EXPECT_FALSE(ParseWorkloadJsonLines("{\"schema_version\":2,\"event_type\":\"cancel\",\"at_ns\":0,"
                                      "\"request_id\":1}\n")
                   .ok());
  EXPECT_FALSE(ParseWorkloadJsonLines("{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
                                      "\"request_id\":1,\"model_id\":0,\"token_ids\":[],"
                                      "\"max_output_tokens\":1}\n")
                   .ok());
  EXPECT_FALSE(ParseWorkloadJsonLines("{\"schema_version\":1,\"event_type\":\"cancel\",\"at_ns\":2,"
                                      "\"request_id\":1}\n"
                                      "{\"schema_version\":1,\"event_type\":\"cancel\",\"at_ns\":1,"
                                      "\"request_id\":1}\n")
                   .ok());
}

TEST(SimulatorFailureTest, ReplayRejectsTruncationUnknownVersionTypeAndMissingFields) {
  EXPECT_FALSE(CheckReplayJsonLines("").ok());
  EXPECT_FALSE(
      CheckReplayJsonLines("{\"schema_version\":2,\"record_type\":\"header\",\"ordinal\":0}\n"
                           "{\"schema_version\":1,\"record_type\":\"footer\",\"ordinal\":1}\n")
          .ok());
  EXPECT_FALSE(
      CheckReplayJsonLines("{\"schema_version\":1,\"record_type\":\"unknown\",\"ordinal\":0}\n")
          .ok());
  EXPECT_FALSE(
      CheckReplayJsonLines("{\"schema_version\":1,\"record_type\":\"header\",\"ordinal\":0}\n"
                           "{\"schema_version\":1,\"record_type\":\"footer\",\"ordinal\":1}\n")
          .ok());
}

TEST(SimulatorFailureTest, InvalidDocumentedEnvironmentValueIsNotIgnored) {
  ASSERT_EQ(setenv("INFERX_MAX_ACTIVE_SEQUENCES", "not-a-number", 1), 0);
  const auto values = config::ReadConfigEnvironment();
  EXPECT_FALSE(values.ok());
  ASSERT_EQ(unsetenv("INFERX_MAX_ACTIVE_SEQUENCES"), 0);
}

TEST(SimulatorFailureTest, WorkloadCapacityBoundFailsBeforePartialLoad) {
  config::ParsedConfig parsed;
  parsed.SetMaxSimulationEvents(1, config::ConfigSource::kCommandLine);
  auto config = config::EngineConfig::Validate(
      parsed, {}, {.max_context_tokens = TokenCount(32768), .accepts_text = false});
  ASSERT_TRUE(config.ok()) << config.status();
  auto workload = ParseWorkloadJsonLines(
      "{\"schema_version\":1,\"event_type\":\"submit\",\"at_ns\":0,"
      "\"request_id\":1,\"model_id\":0,\"token_ids\":[1],\"max_output_tokens\":1}\n");
  ASSERT_TRUE(workload.ok()) << workload.status();
  InMemoryReplaySink replay(ReplayBufferLimits{.max_bytes = 1U << 20U, .max_records = 64});
  auto simulator = EngineSimulator::Create(
      *config, {.max_context_tokens = TokenCount(32768), .accepts_text = false}, replay);
  ASSERT_TRUE(simulator.ok()) << simulator.status();
  const absl::Status loaded = (*simulator)->LoadWorkload(*workload);
  EXPECT_FALSE(loaded.ok());
  EXPECT_EQ(loaded.code(), absl::StatusCode::kResourceExhausted);
  EXPECT_TRUE((*simulator)->responses().empty());
}

}  // namespace
}  // namespace inferx::simulator
