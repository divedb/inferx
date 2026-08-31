#include "inferx/config/engine_config.h"

#include <algorithm>
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

  EngineConfig effective;
  effective.values_ = parsed;
  return effective;
}

std::string EngineConfig::CanonicalJson() const {
  // Schema version first, then lexicographic field order, decimal integers,
  // no insignificant whitespace (m1.md section 8.3).
  std::string out = "{\"schema_version\":1";
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
  out += "}";
  return out;
}

}  // namespace inferx::config
