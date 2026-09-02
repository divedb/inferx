// Configuration pipeline tests (ADR 0011).
#include <gtest/gtest.h>

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/token.h"
#include "inferx/config/config_loader.h"
#include "inferx/config/engine_config.h"
#include "inferx/config/parser_limits.h"

namespace {

using inferx::TokenCount;
using inferx::config::BuildCapabilities;
using inferx::config::ConfigSource;
using inferx::config::EngineConfig;
using inferx::config::FieldValues;
using inferx::config::kMaxConfigFileBytes;
using inferx::config::LoadConfig;
using inferx::config::ModelCapabilities;
using inferx::config::ParseConfigInteger;
using inferx::config::ParseConfigJson;
using inferx::config::ParsedConfig;

ModelCapabilities DefaultModel() {
  return ModelCapabilities{.max_context_tokens = TokenCount(131072)};
}

FieldValues V(const std::string& json) {
  auto parsed = ParseConfigJson(json);
  if (!parsed.ok()) {
    ADD_FAILURE() << parsed.status().message();
    return {};
  }
  return *parsed;
}

TEST(ConfigJsonTest, ParsesFlatIntegerObject) {
  const FieldValues values = V(R"({"max_active_sequences": 7, "plan_buffer_slots": 2})");
  EXPECT_EQ(values.at("max_active_sequences"), 7u);
  EXPECT_EQ(values.at("plan_buffer_slots"), 2u);
}

TEST(ConfigJsonTest, UnknownFieldRejected) {
  const auto rejected = ParseConfigJson(R"({"max_prompt_tokenz": 1})");
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(rejected.status().message().find("max_prompt_tokenz"), std::string::npos);
}

TEST(ConfigJsonTest, DuplicateKeyRejected) {
  const auto rejected = ParseConfigJson(R"({"max_output_tokens": 1, "max_output_tokens": 2})");
  ASSERT_FALSE(rejected.ok());
  EXPECT_NE(rejected.status().message().find("duplicate"), std::string::npos);
}

TEST(ConfigJsonTest, NonIntegerValueRejected) {
  EXPECT_FALSE(ParseConfigJson(R"({"max_output_tokens": 1.5})").ok());
  EXPECT_FALSE(ParseConfigJson(R"({"max_output_tokens": "8"})").ok());
  EXPECT_FALSE(ParseConfigJson(R"({"max_output_tokens": true})").ok());
  EXPECT_FALSE(ParseConfigJson(R"({"max_output_tokens": -8})").ok());
}

TEST(ConfigJsonTest, InvalidUtf8AndSyntaxRejected) {
  const auto invalid_utf8 = ParseConfigJson(std::string("{\"max_output_tokens\":1}\xFF\xFE"));
  EXPECT_FALSE(invalid_utf8.ok());
  EXPECT_FALSE(ParseConfigJson("{\"max_output_tokens\":").ok());
  EXPECT_FALSE(ParseConfigJson("[1,2]").ok());  // top level must be an object
}

TEST(ConfigJsonTest, SizeLimit) {
  std::string big = "{\"max_output_tokens\":1,\"padding\":\"";
  big += std::string(kMaxConfigFileBytes, 'x');
  big += "\"}";
  const auto rejected = ParseConfigJson(big);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(rejected.status().message().find("exceeds limit"), std::string::npos);
}

TEST(ConfigIntegerTest, StrictParsing) {
  EXPECT_EQ(ParseConfigInteger("42", "f").value(), 42u);
  EXPECT_EQ(ParseConfigInteger("0", "f").value(), 0u);
  EXPECT_FALSE(ParseConfigInteger("", "f").ok());
  EXPECT_FALSE(ParseConfigInteger(" 42", "f").ok());
  EXPECT_FALSE(ParseConfigInteger("42 ", "f").ok());
  EXPECT_FALSE(ParseConfigInteger("-1", "f").ok());
  EXPECT_FALSE(ParseConfigInteger("0x10", "f").ok());
  EXPECT_FALSE(ParseConfigInteger("18446744073709551616", "f").ok());  // 2^64
}

TEST(ConfigLoadTest, DefaultsAndPrecedence) {
  const auto loaded =
      LoadConfig(V("{\"max_output_tokens\": 9}"), FieldValues{{"max_output_tokens", 10}},
                 FieldValues{{"max_output_tokens", 11}});
  ASSERT_TRUE(loaded.ok()) << loaded.status().message();
  EXPECT_EQ(loaded->MaxOutputTokens.value, 11u);
  EXPECT_EQ(loaded->MaxOutputTokens.source, ConfigSource::kCommandLine);
  EXPECT_EQ(loaded->MaxQueuedRequests.value, 4096u);
  EXPECT_EQ(loaded->MaxQueuedRequests.source, ConfigSource::kDefault);
}

TEST(ConfigLoadTest, LaterLayerDoesNotMaskInvalidEarlierValue) {
  // An invalid value is an error even if a later source would replace it.
  const auto rejected = LoadConfig(FieldValues{{"max_queued_requests", 2000000}},  // > 1e6
                                   std::nullopt, FieldValues{{"max_queued_requests", 128}});
  // LoadConfig itself accepts (range checks live in validation); validation
  // below must reject. This asserts the layering contract precisely:
  // loading never drops values, validation never sees them masked.
  ASSERT_TRUE(rejected.ok());
  const auto validated = EngineConfig::Validate(*rejected, BuildCapabilities{}, DefaultModel());
  ASSERT_FALSE(validated.ok());
}

TEST(ConfigLoadTest, UnknownOverlayFieldRejectedWithSource) {
  const auto rejected =
      LoadConfig(std::nullopt, FieldValues{{"max_output_tokenz", 1}}, std::nullopt);
  ASSERT_FALSE(rejected.ok());
  EXPECT_NE(rejected.status().message().find("environment.max_output_tokenz"), std::string::npos);
}

TEST(ConfigValidateTest, DefaultsValidate) {
  const auto effective =
      EngineConfig::Validate(ParsedConfig{}, BuildCapabilities{}, DefaultModel());
  ASSERT_TRUE(effective.ok()) << effective.status().message();
  EXPECT_EQ(effective->MaxQueuedRequests(), 4096u);
}

TEST(ConfigValidateTest, RangeErrors) {
  ParsedConfig zero_sequences;
  zero_sequences.SetMaxActiveSequences(0, ConfigSource::kFile);
  const auto rejected = EngineConfig::Validate(zero_sequences, BuildCapabilities{}, DefaultModel());
  ASSERT_FALSE(rejected.ok());
  EXPECT_NE(rejected.status().message().find("config.max_active_sequences"), std::string::npos);
}

TEST(ConfigValidateTest, CrossFieldRules) {
  ParsedConfig too_many_active;
  too_many_active.SetMaxActiveSequences(5000, ConfigSource::kFile);
  too_many_active.SetMaxQueuedRequests(4000, ConfigSource::kFile);
  EXPECT_FALSE(EngineConfig::Validate(too_many_active, BuildCapabilities{}, DefaultModel()).ok());

  ParsedConfig sequences_step_too_big;
  sequences_step_too_big.SetMaxActiveSequences(1, ConfigSource::kFile);
  sequences_step_too_big.SetMaxSequencesPerStep(2, ConfigSource::kFile);
  EXPECT_FALSE(
      EngineConfig::Validate(sequences_step_too_big, BuildCapabilities{}, DefaultModel()).ok());

  ParsedConfig prompt_exceeds_step;
  prompt_exceeds_step.SetMaxPromptTokens(5000, ConfigSource::kFile);
  prompt_exceeds_step.SetMaxScheduledTokensPerStep(4096, ConfigSource::kFile);
  EXPECT_FALSE(
      EngineConfig::Validate(prompt_exceeds_step, BuildCapabilities{}, DefaultModel()).ok());
}

TEST(ConfigValidateTest, ModelCapabilityMismatch) {
  // default max_model_tokens (4352) exceeds this capability:
  ModelCapabilities small{.max_context_tokens = TokenCount(2048)};
  const auto rejected = EngineConfig::Validate(ParsedConfig{}, BuildCapabilities{}, small);
  ASSERT_FALSE(rejected.ok());
  EXPECT_NE(rejected.status().message().find("config.max_model_tokens"), std::string::npos);
}

TEST(ConfigValidateTest, LatencyOverflowRejected) {
  ParsedConfig huge;
  huge.SetMaxScheduledTokensPerStep(2147483647, ConfigSource::kFile);
  // 2^31-1 tokens * 1e10 ns/token exceeds uint64 on purpose.
  huge.SetFakePrefillLatencyPerTokenNs(10000000000ull, ConfigSource::kFile);
  const auto rejected = EngineConfig::Validate(huge, BuildCapabilities{}, DefaultModel());
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kOutOfRange);
}

bool CanonicalShapeOk(const std::string& canonical);

TEST(ConfigCanonicalTest, ByteIdenticalAcrossInputOrderAndLayers) {
  const auto from_file = LoadConfig(
      V(R"({"max_output_tokens": 9, "max_active_sequences": 64, "plan_buffer_slots": 2})"),
      std::nullopt, std::nullopt);
  const auto from_layers = LoadConfig(V(R"({"plan_buffer_slots": 2, "max_active_sequences": 64})"),
                                      FieldValues{{"max_output_tokens", 9}}, std::nullopt);
  ASSERT_TRUE(from_file.ok());
  ASSERT_TRUE(from_layers.ok());

  const auto effective_a = EngineConfig::Validate(*from_file, BuildCapabilities{}, DefaultModel());
  const auto effective_b =
      EngineConfig::Validate(*from_layers, BuildCapabilities{}, DefaultModel());
  ASSERT_TRUE(effective_a.ok());
  ASSERT_TRUE(effective_b.ok());
  EXPECT_EQ(effective_a->CanonicalJson(), effective_b->CanonicalJson());

  // Exact canonical shape: version first, lexicographic, no whitespace.
  const std::string canonical = effective_a->CanonicalJson();
  EXPECT_TRUE(CanonicalShapeOk(canonical));
}

bool CanonicalShapeOk(const std::string& canonical) {
  if (canonical.substr(0, 19) != "{\"schema_version\":1") {
    return false;
  }
  if (canonical.find(' ') != std::string::npos) {
    return false;
  }
  // max_active_sequences sorts before max_model_tokens sorts before
  // max_output_tokens — assert the first three fields' order.
  const size_t active = canonical.find("\"max_active_sequences\":");
  const size_t model = canonical.find("\"max_model_tokens\":");
  const size_t output = canonical.find("\"max_output_tokens\":");
  return active < model && model < output;
}

}  // namespace
