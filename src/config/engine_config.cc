#include "inferx/config/engine_config.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"
#include "inferx/base/status_macros.h"
#include "inferx/config/parser_limits.h"

namespace inferx::config {

namespace {

absl::Status FieldError(absl::string_view field, absl::string_view message) {
  return inferx::FieldError(absl::StatusCode::kInvalidArgument, "config", field, message);
}

struct Range {
  absl::string_view field;
  uint64_t value;
  uint64_t min;
  uint64_t max;  // inclusive; 0 means "no upper check beyond 64-bit"
};

// Latency products must not overflow when the fake executor scales them by
// worst-case token/sequence budgets.
absl::Status CheckLatencyOverflow(const ParsedConfig& values) {
  const uint64_t max_tokens = values.MaxScheduledTokensPerStep.value;
  const uint64_t max_sequences = values.MaxSequencesPerStep.value;
  absl::StatusOr<uint64_t> prefill =
      CheckedMul(values.FakePrefillLatencyPerTokenNs.value, max_tokens,
                 "config.fake_prefill_latency_per_token_ns");
  if (!prefill.ok()) {
    return prefill.status();
  }
  absl::StatusOr<uint64_t> decode =
      CheckedMul(values.FakeDecodeLatencyPerSequenceNs.value, max_sequences,
                 "config.fake_decode_latency_per_sequence_ns");
  if (!decode.ok()) {
    return decode.status();
  }
  absl::StatusOr<uint64_t> scaled = CheckedAdd(*prefill, *decode, "config.fake_latency");
  if (!scaled.ok()) {
    return scaled.status();
  }
  absl::StatusOr<uint64_t> total =
      CheckedAdd(values.FakeBaseLatencyNs.value, *scaled, "config.fake_latency");
  if (!total.ok()) {
    return total.status();
  }
  return absl::OkStatus();
}

}  // namespace

std::vector<std::string> CanonicalFieldOrder() {
  std::vector<std::string> names;
#define INFERX_COLLECT(camel, json_name, default_value) names.emplace_back(json_name);
  INFERX_CONFIG_FIELDS(INFERX_COLLECT)
  INFERX_CUDA_CONFIG_FIELDS(INFERX_COLLECT)
#undef INFERX_COLLECT
  std::sort(names.begin(), names.end());
  return names;
}

absl::StatusOr<EngineConfig> EngineConfig::Validate(const ParsedConfig& parsed,
                                                    const BuildCapabilities& build,
                                                    const ModelCapabilities& model) {
  // Independent per-field range errors, reported in stable order.
  const std::vector<Range> ranges = {
      {"max_active_sequences", parsed.MaxActiveSequences.value, 1, 1000000},
      {"max_model_tokens", parsed.MaxModelTokens.value, 2, 2147483647},
      {"max_output_tokens", parsed.MaxOutputTokens.value, 1, 2147483647},
      {"max_prompt_tokens", parsed.MaxPromptTokens.value, 1, 2147483647},
      {"max_queued_requests", parsed.MaxQueuedRequests.value, 1, 1000000},
      {"max_scheduled_tokens_per_step", parsed.MaxScheduledTokensPerStep.value, 1, 2147483647},
      {"max_sequences_per_step", parsed.MaxSequencesPerStep.value, 1, 16384},
      {"max_simulation_events", parsed.MaxSimulationEvents.value, 1, 100000000},
      {"plan_buffer_slots", parsed.PlanBufferSlots.value, 1, 64},
      {"response_channel_capacity", parsed.ResponseChannelCapacity.value, 1, 0},
      {"simulated_kv_token_capacity", parsed.SimulatedKvTokenCapacity.value, 1, 0},
      {"submission_channel_capacity", parsed.SubmissionChannelCapacity.value, 1, 0},
      {"fake_base_latency_ns", parsed.FakeBaseLatencyNs.value, 1, 0},
      {"fake_prefill_latency_per_token_ns", parsed.FakePrefillLatencyPerTokenNs.value, 0, 0},
      {"fake_decode_latency_per_sequence_ns", parsed.FakeDecodeLatencyPerSequenceNs.value, 0, 0},
      {"cuda.device_budget_bytes", parsed.CudaDeviceBudgetBytes.value, 0, 0},
      {"cuda.device_id", parsed.CudaDeviceId.value, 0, std::numeric_limits<uint32_t>::max()},
      {"cuda.device_reserve_bytes", parsed.CudaDeviceReserveBytes.value, 0, 0},
      {"cuda.enable_transfer_stream", parsed.CudaEnableTransferStream.value, 0, 1},
      {"cuda.enabled", parsed.CudaEnabled.value, 0, 1},
      {"cuda.event_pool_slots", parsed.CudaEventPoolSlots.value, 8, 65536},
      {"cuda.metadata_ring_slots", parsed.CudaMetadataRingSlots.value, 2, 64},
      {"cuda.metadata_slot_bytes", parsed.CudaMetadataSlotBytes.value, 4096, 16777216},
      {"cuda.pinned_budget_bytes", parsed.CudaPinnedBudgetBytes.value, 1048576, 4294967296ULL},
      {"cuda.staging_pool_slots", parsed.CudaStagingPoolSlots.value, 2, 256},
      {"cuda.staging_slot_bytes", parsed.CudaStagingSlotBytes.value, 4096, 67108864},
      {"cuda.timing_event_slots", parsed.CudaTimingEventSlots.value, 2, 256},
      {"cuda.workspace_bytes_per_slot", parsed.CudaWorkspaceBytesPerSlot.value, 1048576,
       4294967296ULL},
      {"cuda.workspace_slots", parsed.CudaWorkspaceSlots.value, 1, 64},
  };

  for (const Range& range : ranges) {
    if (range.value < range.min || (range.max != 0 && range.value > range.max)) {
      return FieldError(range.field,
                        absl::StrCat("value must be in [", range.min, ", ",
                                     range.max == 0 ? "2^64-1" : absl::StrCat(range.max), "]; got ",
                                     range.value));
    }
  }

  // Cross-field rules (m1.md section 8.2).
  if (parsed.MaxActiveSequences.value > parsed.MaxQueuedRequests.value) {
    return FieldError("max_active_sequences", "must be <= max_queued_requests");
  }
  const uint64_t sequences_cap =
      std::min<uint64_t>(parsed.MaxActiveSequences.value, kMaxSequencesPerStepHardCap);
  if (parsed.MaxSequencesPerStep.value > sequences_cap) {
    return FieldError("max_sequences_per_step", "must be <= min(max_active_sequences, 16384)");
  }
  if (parsed.MaxPromptTokens.value > parsed.MaxScheduledTokensPerStep.value) {
    return FieldError("max_prompt_tokens",
                      "must be <= max_scheduled_tokens_per_step in M1 "
                      "(chunking is M8)");
  }
  if (parsed.MaxModelTokens.value > static_cast<uint64_t>(model.max_context_tokens.value())) {
    return FieldError("max_model_tokens", absl::StrCat("must not exceed model capability ",
                                                       model.max_context_tokens.value()));
  }
  if (!build.simulator) {
    return FieldError("build_capabilities", "M1 configuration requires the simulator build");
  }
  if (absl::Status overflow = CheckLatencyOverflow(parsed); !overflow.ok()) {
    return overflow;
  }

  const bool has_cuda_section =
#define INFERX_CUDA_PRESENT(camel, json_name, default_value) \
  parsed.camel.source != ConfigSource::kDefault ||
      INFERX_CUDA_CONFIG_FIELDS(INFERX_CUDA_PRESENT)
#undef INFERX_CUDA_PRESENT
          false;
  if (parsed.CudaEnabled.value != 0 && !build.cuda) {
    return FieldError("cuda.enabled", "true requires an INFERX_ENABLE_CUDA build");
  }
  if (has_cuda_section) {
    if (!std::has_single_bit(parsed.CudaMetadataSlotBytes.value)) {
      return FieldError("cuda.metadata_slot_bytes", "must be a power of two");
    }
    if (!std::has_single_bit(parsed.CudaStagingSlotBytes.value)) {
      return FieldError("cuda.staging_slot_bytes", "must be a power of two");
    }
    if ((parsed.CudaStagingPoolSlots.value & 1U) != 0) {
      return FieldError("cuda.staging_pool_slots", "must be even");
    }
    if (parsed.CudaWorkspaceSlots.value < parsed.PlanBufferSlots.value) {
      return FieldError("cuda.workspace_slots", "must cover configured in-flight plan buffers");
    }
    const uint64_t max_test_inflight =
        std::min({parsed.CudaMetadataRingSlots.value, parsed.CudaWorkspaceSlots.value,
                  parsed.CudaStagingPoolSlots.value / 2});
    absl::StatusOr<uint64_t> completion_events =
        CheckedMul(max_test_inflight, uint64_t{2}, "config.cuda.event_pool_slots");
    if (!completion_events.ok()) {
      return completion_events.status();
    }
    absl::StatusOr<uint64_t> required_events = CheckedAdd(
        parsed.CudaMetadataRingSlots.value, *completion_events, "config.cuda.event_pool_slots");
    if (!required_events.ok()) {
      return required_events.status();
    }
    if (parsed.CudaEventPoolSlots.value < *required_events) {
      return FieldError("cuda.event_pool_slots",
                        "must cover metadata uploads plus two pipeline events per in-flight test");
    }
    absl::StatusOr<uint64_t> metadata_total =
        CheckedMul(parsed.CudaMetadataRingSlots.value, parsed.CudaMetadataSlotBytes.value,
                   "config.cuda.metadata_bytes");
    if (!metadata_total.ok()) return metadata_total.status();
    absl::StatusOr<uint64_t> staging_total =
        CheckedMul(parsed.CudaStagingPoolSlots.value, parsed.CudaStagingSlotBytes.value,
                   "config.cuda.staging_bytes");
    if (!staging_total.ok()) return staging_total.status();
    absl::StatusOr<uint64_t> workspace_total =
        CheckedMul(parsed.CudaWorkspaceSlots.value, parsed.CudaWorkspaceBytesPerSlot.value,
                   "config.cuda.workspace_bytes");
    if (!workspace_total.ok()) return workspace_total.status();
    if (*staging_total > parsed.CudaPinnedBudgetBytes.value ||
        *metadata_total > parsed.CudaPinnedBudgetBytes.value - *staging_total) {
      return FieldError("cuda.pinned_budget_bytes", "must cover staging and pinned metadata pools");
    }
    if (parsed.CudaDeviceBudgetBytes.value != 0 &&
        (*metadata_total > parsed.CudaDeviceBudgetBytes.value ||
         *workspace_total > parsed.CudaDeviceBudgetBytes.value - *metadata_total)) {
      return FieldError("cuda.device_budget_bytes",
                        "must cover device metadata and workspace pools");
    }
  }

  EngineConfig effective;
  effective.values_ = parsed;
  effective.has_cuda_section_ = has_cuda_section;
  return effective;
}

std::string EngineConfig::CanonicalJson() const {
  // Schema version first, then lexicographic field order, decimal integers,
  // no insignificant whitespace (m1.md section 8.3).
  std::string out = has_cuda_section_ ? "{\"schema_version\":2" : "{\"schema_version\":1";
  std::vector<std::pair<std::string, uint64_t>> fields;
#define INFERX_PAIR(camel, json_name, default_value) \
  fields.emplace_back(json_name, values_.camel.value);
  INFERX_CONFIG_FIELDS(INFERX_PAIR)
#undef INFERX_PAIR
  std::sort(fields.begin(), fields.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& [name, value] : fields) {
    out += ",\"";
    out += name;
    out += "\":";
    out += absl::StrCat(value);
  }
  if (has_cuda_section_) {
    out += ",\"cuda\":{";
    std::vector<std::pair<std::string, uint64_t>> cuda_fields;
#define INFERX_CUDA_PAIR(camel, json_name, default_value) \
  cuda_fields.emplace_back(std::string(json_name).substr(5), values_.camel.value);
    INFERX_CUDA_CONFIG_FIELDS(INFERX_CUDA_PAIR)
#undef INFERX_CUDA_PAIR
    std::sort(cuda_fields.begin(), cuda_fields.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    bool first = true;
    for (const auto& [name, value] : cuda_fields) {
      if (!first) out += ",";
      first = false;
      out += "\"" + name + "\":";
      if (name == "enabled" || name == "enable_transfer_stream") {
        out += value == 0 ? "false" : "true";
      } else if (name == "device_budget_bytes" && value == 0) {
        out += "null";
      } else {
        out += absl::StrCat(value);
      }
    }
    out += "}";
  }
  out += "}";
  return out;
}

}  // namespace inferx::config
