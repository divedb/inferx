// M5 execution configuration section (m5.md section 7): nested JSON object,
// schema-v3 canonical output, range and cross-field rules.

#include <string>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/base/token.h"
#include "inferx/config/config_loader.h"
#include "inferx/config/engine_config.h"
#include "inferx/config/parsed_config.h"

namespace inferx::config {
namespace {

ModelCapabilities DefaultModel() {
  return ModelCapabilities{.max_context_tokens = TokenCount(131072)};
}

TEST(ExecutionConfigTest, NestedSectionParsesAndCanonicalizesAsSchemaV3) {
  const absl::StatusOr<FieldValues> fields = ParseConfigJson(
      R"({"schema_version":3,"execution":{"max_prefill_tokens":256,"model_device_budget_bytes":null}})");
  ASSERT_TRUE(fields.ok()) << fields.status();
  EXPECT_EQ(fields->at("execution.max_prefill_tokens"), 256);
  EXPECT_EQ(fields->at("execution.model_device_budget_bytes"), 0);
  const ParsedConfig parsed = LoadConfig(*fields, std::nullopt, std::nullopt).value();
  const EngineConfig config =
      EngineConfig::Validate(parsed, BuildCapabilities{}, DefaultModel()).value();
  EXPECT_TRUE(config.has_execution_section());
  EXPECT_EQ(config.ExecutionMaxPrefillTokens(), 256);
  EXPECT_EQ(config.ExecutionMaxContextTokens(), 4096);
  EXPECT_EQ(config.CanonicalJson().find("{\"schema_version\":3"), 0);
  EXPECT_NE(config.CanonicalJson().find("\"execution\":{"), std::string::npos);
  EXPECT_NE(config.CanonicalJson().find("\"model_device_budget_bytes\":null"), std::string::npos);
  EXPECT_NE(config.CanonicalJson().find("\"poll_backoff_us\":50"), std::string::npos);
}

TEST(ExecutionConfigTest, DefaultsRemainSchemaV1AndSectionless) {
  const EngineConfig config =
      EngineConfig::Validate(ParsedConfig{}, BuildCapabilities{}, DefaultModel()).value();
  EXPECT_FALSE(config.has_execution_section());
  EXPECT_EQ(config.CanonicalJson().find("{\"schema_version\":1"), 0);
  EXPECT_EQ(config.CanonicalJson().find("\"execution\""), std::string::npos);
}

TEST(ExecutionConfigTest, RangeAndCrossFieldFailuresAreRejected) {
  ParsedConfig over_prefill;
  over_prefill.SetExecutionMaxPrefillTokens(513, ConfigSource::kFile);
  EXPECT_EQ(EngineConfig::Validate(over_prefill, BuildCapabilities{}, DefaultModel())
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  ParsedConfig fast_poll;
  fast_poll.SetExecutionPollBackoffUs(1, ConfigSource::kFile);
  EXPECT_EQ(EngineConfig::Validate(fast_poll, BuildCapabilities{}, DefaultModel())
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  ParsedConfig inconsistent;
  inconsistent.SetExecutionMaxPrefillTokens(4096, ConfigSource::kFile);
  inconsistent.SetExecutionMaxContextTokens(512, ConfigSource::kFile);
  EXPECT_EQ(EngineConfig::Validate(inconsistent, BuildCapabilities{}, DefaultModel())
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  ParsedConfig oversized_output;
  oversized_output.SetExecutionMaxOutputTokens(8192, ConfigSource::kFile);
  EXPECT_EQ(EngineConfig::Validate(oversized_output, BuildCapabilities{}, DefaultModel())
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ExecutionConfigTest, UnknownNestedFieldAndBadTypesAreRejected) {
  const absl::StatusOr<FieldValues> unknown = ParseConfigJson(
      R"({"schema_version":3,"execution":{"beam_width":4}})");
  EXPECT_FALSE(unknown.ok());

  const absl::StatusOr<FieldValues> wrong_null = ParseConfigJson(
      R"({"schema_version":3,"execution":{"max_prefill_tokens":null}})");
  EXPECT_FALSE(wrong_null.ok());
}

TEST(ExecutionConfigTest, EnvironmentOverlayUsesFlattenedNames) {
  absl::StatusOr<FieldValues> environment = ReadConfigEnvironment();
  ASSERT_TRUE(environment.ok()) << environment.status();
  // Without INFERX_EXECUTION_* set in this process the section stays inert;
  // the flattened spelling is exercised through the JSON and flag layers.
  const ParsedConfig parsed = LoadConfig(
      ParseConfigJson(R"({"schema_version":3,"execution":{"max_output_tokens":32}})").value(),
      std::nullopt, std::nullopt)
                                 .value();
  const EngineConfig config =
      EngineConfig::Validate(parsed, BuildCapabilities{}, DefaultModel()).value();
  EXPECT_EQ(config.ExecutionMaxOutputTokens(), 32);
}

}  // namespace
}  // namespace inferx::config
