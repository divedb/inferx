// Parsed configuration: one value plus provenance per schema field
// (m1.md section 8.1). Defaults < JSON file < environment < CLI; later
// sources override only fields they contain, and an invalid value in any
// source is an error even when a later source would replace it. Validation
// happens later, in ValidateConfig; this type carries parsed input only.

#ifndef INFERX_CONFIG_PARSED_CONFIG_H_
#define INFERX_CONFIG_PARSED_CONFIG_H_

#include <cstdint>

namespace inferx::config {

enum class ConfigSource : uint8_t {
  kDefault,
  kFile,
  kEnvironment,
  kCommandLine,
};

struct SourcedValue {
  uint64_t value = 0;
  ConfigSource source = ConfigSource::kDefault;
};

// Every M1 schema field is an unsigned integer. The X-macro list is the one
// registry (member, JSON/CLI spelling, default); ranges and cross-field rules
// live in engine_config.cc; parsing and overlay application live in
// config_loader.cc. Canonical output sorts lexicographically by field name,
// independent of this declaration order.
#define INFERX_CONFIG_FIELDS(X)                                             \
  X(MaxActiveSequences, "max_active_sequences", 256)                        \
  X(MaxModelTokens, "max_model_tokens", 4352)                               \
  X(MaxOutputTokens, "max_output_tokens", 256)                              \
  X(MaxPromptTokens, "max_prompt_tokens", 4096)                             \
  X(MaxQueuedRequests, "max_queued_requests", 4096)                         \
  X(MaxScheduledTokensPerStep, "max_scheduled_tokens_per_step", 4096)       \
  X(MaxSequencesPerStep, "max_sequences_per_step", 32)                      \
  X(MaxSimulationEvents, "max_simulation_events", 10000000)                 \
  X(PlanBufferSlots, "plan_buffer_slots", 1)                                \
  X(ResponseChannelCapacity, "response_channel_capacity", 4096)             \
  X(SimulatedKvTokenCapacity, "simulated_kv_token_capacity", 1114112)       \
  X(SubmissionChannelCapacity, "submission_channel_capacity", 4096)         \
  X(FakeBaseLatencyNs, "fake_base_latency_ns", 1000)                        \
  X(FakePrefillLatencyPerTokenNs, "fake_prefill_latency_per_token_ns", 100) \
  X(FakeDecodeLatencyPerSequenceNs, "fake_decode_latency_per_sequence_ns", 100)

#define INFERX_CONFIG_MEMBER(camel, json_name, default_value)                                    \
  SourcedValue camel = SourcedValue{(default_value), ConfigSource::kDefault};                    \
  void Set##camel /* NOLINT(bugprone-macro-parentheses): paste target */ (uint64_t value,        \
                                                                          ConfigSource source) { \
    (camel) = SourcedValue{(value), (source)};                                                   \
  }

struct ParsedConfig {
  INFERX_CONFIG_FIELDS(INFERX_CONFIG_MEMBER)
};

#undef INFERX_CONFIG_MEMBER
// INFERX_CONFIG_FIELDS stays defined: it is the single schema registry
// reused by engine_config.h accessors and canonical writers.

}  // namespace inferx::config

#endif  // INFERX_CONFIG_PARSED_CONFIG_H_
