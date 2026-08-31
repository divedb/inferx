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

TEST(M2CudaConfigTest, NestedSectionParsesAndCanonicalizesAsSchemaV2) {
  const absl::StatusOr<FieldValues> fields = ParseConfigJson(
      R"({"schema_version":2,"cuda":{"enabled":false,"device_budget_bytes":null,"event_pool_slots":64}})");
  ASSERT_TRUE(fields.ok()) << fields.status();
  EXPECT_EQ(fields->at("cuda.enabled"), 0);
  EXPECT_EQ(fields->at("cuda.device_budget_bytes"), 0);
  EXPECT_EQ(fields->at("cuda.event_pool_slots"), 64);
  const ParsedConfig parsed = LoadConfig(*fields, std::nullopt, std::nullopt).value();
  const EngineConfig config =
      EngineConfig::Validate(parsed, BuildCapabilities{}, DefaultModel()).value();
  EXPECT_TRUE(config.has_cuda_section());
  EXPECT_EQ(config.CudaEventPoolSlots(), 64);
  EXPECT_EQ(config.CanonicalJson().find("{\"schema_version\":2"), 0);
  EXPECT_NE(config.CanonicalJson().find("\"cuda\":{"), std::string::npos);
  EXPECT_NE(config.CanonicalJson().find("\"device_budget_bytes\":null"), std::string::npos);
}

TEST(M2CudaConfigTest, SchemaV1DefaultsRemainByteStable) {
  const EngineConfig config =
      EngineConfig::Validate(ParsedConfig{}, BuildCapabilities{}, DefaultModel()).value();
  EXPECT_FALSE(config.has_cuda_section());
  EXPECT_EQ(config.CanonicalJson().find("{\"schema_version\":1"), 0);
  EXPECT_EQ(config.CanonicalJson().find("\"cuda\""), std::string::npos);
}

TEST(M2CudaConfigTest, SchemaV1OverridesDoNotActivateCudaCrossFieldRules) {
  ParsedConfig parsed;
  parsed.SetPlanBufferSlots(8, ConfigSource::kFile);

  const absl::StatusOr<EngineConfig> config =
      EngineConfig::Validate(parsed, BuildCapabilities{}, DefaultModel());
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_FALSE(config->has_cuda_section());
  EXPECT_EQ(config->PlanBufferSlots(), 8);
  EXPECT_EQ(config->CanonicalJson().find("{\"schema_version\":1"), 0);
  EXPECT_EQ(config->CanonicalJson().find("\"cuda\""), std::string::npos);
}

TEST(M2CudaConfigTest, CapabilityAndCrossFieldFailuresAreRejected) {
  ParsedConfig enabled;
  enabled.SetCudaEnabled(1, ConfigSource::kFile);
  EXPECT_EQ(EngineConfig::Validate(enabled, BuildCapabilities{}, DefaultModel()).status().code(),
            absl::StatusCode::kInvalidArgument);

  ParsedConfig event_shortage;
  event_shortage.SetCudaEventPoolSlots(8, ConfigSource::kFile);
  event_shortage.SetCudaMetadataRingSlots(4, ConfigSource::kFile);
  event_shortage.SetCudaWorkspaceSlots(4, ConfigSource::kFile);
  event_shortage.SetCudaStagingPoolSlots(8, ConfigSource::kFile);
  EXPECT_FALSE(EngineConfig::Validate(event_shortage, BuildCapabilities{}, DefaultModel()).ok());

  ParsedConfig device_budget_shortage;
  device_budget_shortage.SetCudaDeviceBudgetBytes(1048576, ConfigSource::kFile);
  const absl::Status device_budget_status =
      EngineConfig::Validate(device_budget_shortage, BuildCapabilities{}, DefaultModel()).status();
  EXPECT_EQ(device_budget_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(device_budget_status.message().find("cuda.device_budget_bytes"), std::string::npos);
}

TEST(M2CudaConfigTest, JsonTypesRemainStrict) {
  EXPECT_FALSE(ParseConfigJson(R"({"cuda":{"enabled":1}})").ok());
  EXPECT_FALSE(ParseConfigJson(R"({"cuda":{"event_pool_slots":true}})").ok());
  EXPECT_FALSE(ParseConfigJson(R"({"cuda":{"event_pool_slots":null}})").ok());
  EXPECT_FALSE(ParseConfigJson(R"({"schema_version":3})").ok());
}

}  // namespace
}  // namespace inferx::config
