// Parsed configuration: one value plus provenance per schema field.
// Defaults < JSON file < environment < CLI; later
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

// Every base schema field is an unsigned integer. The X-macro list is the one
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

// CUDA fields use flattened internal names while JSON serialization keeps
// the documented nested `cuda` object. DeviceBudgetBytes == 0 denotes null.
#define INFERX_CUDA_CONFIG_FIELDS(X)                                      \
  X(CudaEnabled, "cuda.enabled", 0)                                       \
  X(CudaDeviceId, "cuda.device_id", 0)                                    \
  X(CudaDeviceReserveBytes, "cuda.device_reserve_bytes", 536870912)       \
  X(CudaDeviceBudgetBytes, "cuda.device_budget_bytes", 0)                 \
  X(CudaPinnedBudgetBytes, "cuda.pinned_budget_bytes", 268435456)         \
  X(CudaEventPoolSlots, "cuda.event_pool_slots", 1024)                    \
  X(CudaTimingEventSlots, "cuda.timing_event_slots", 32)                  \
  X(CudaMetadataRingSlots, "cuda.metadata_ring_slots", 3)                 \
  X(CudaMetadataSlotBytes, "cuda.metadata_slot_bytes", 1048576)           \
  X(CudaStagingPoolSlots, "cuda.staging_pool_slots", 4)                   \
  X(CudaStagingSlotBytes, "cuda.staging_slot_bytes", 4194304)             \
  X(CudaWorkspaceSlots, "cuda.workspace_slots", 4)                        \
  X(CudaWorkspaceBytesPerSlot, "cuda.workspace_bytes_per_slot", 16777216) \
  X(CudaEnableTransferStream, "cuda.enable_transfer_stream", 1)

// M5 execution fields (m5.md section 7). Same conventions as the CUDA
// section: flattened internal names, nested `execution` JSON object,
// ModelDeviceBudgetBytes == 0 denotes null.
#define INFERX_EXECUTION_CONFIG_FIELDS(X)                                           \
  X(ExecutionMaxPrefillTokens, "execution.max_prefill_tokens", 512)                 \
  X(ExecutionMaxContextTokens, "execution.max_context_tokens", 4096)                \
  X(ExecutionMaxOutputTokens, "execution.max_output_tokens", 256)                   \
  X(ExecutionModelDeviceBudgetBytes, "execution.model_device_budget_bytes", 0)      \
  X(ExecutionModelHostBudgetBytes, "execution.model_host_budget_bytes", 2147483648) \
  X(ExecutionPollBackoffUs, "execution.poll_backoff_us", 50)

#define INFERX_CONFIG_MEMBER(camel, json_name, default_value)                                    \
  SourcedValue camel = SourcedValue{(default_value), ConfigSource::kDefault};                    \
  void Set##camel /* NOLINT(bugprone-macro-parentheses): paste target */ (uint64_t value,        \
                                                                          ConfigSource source) { \
    (camel) = SourcedValue{(value), (source)};                                                   \
  }

struct ParsedConfig {
  INFERX_CONFIG_FIELDS(INFERX_CONFIG_MEMBER)
  INFERX_CUDA_CONFIG_FIELDS(INFERX_CONFIG_MEMBER)
  INFERX_EXECUTION_CONFIG_FIELDS(INFERX_CONFIG_MEMBER)
};

#undef INFERX_CONFIG_MEMBER
// INFERX_CONFIG_FIELDS stays defined: it is the single schema registry
// reused by engine_config.h accessors and canonical writers.

}  // namespace inferx::config

#endif  // INFERX_CONFIG_PARSED_CONFIG_H_
