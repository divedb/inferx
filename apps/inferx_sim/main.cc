#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"
#include "inferx/config/config_loader.h"
#include "inferx/config/engine_config.h"
#include "inferx/config/parsed_config.h"
#include "inferx/simulator/engine_simulator.h"
#include "inferx/simulator/replay.h"
#include "inferx/simulator/workload.h"

ABSL_FLAG(std::string, config, "", "Path to a schema-v1 JSON configuration file.");
ABSL_FLAG(std::string, workload, "", "Path to a schema-v1 JSONL workload file.");
ABSL_FLAG(std::string, trace, "", "Path to a schema-v1 JSONL replay trace.");
ABSL_FLAG(std::string, output, "", "Path for a newly replayed trace.");
ABSL_FLAG(bool, overwrite, false, "Allow replacing the output trace.");
ABSL_FLAG(bool, verbose, false, "Print the canonical effective configuration.");

ABSL_FLAG(std::optional<uint64_t>, max_active_sequences, std::nullopt,
          "Override max_active_sequences.");
ABSL_FLAG(std::optional<uint64_t>, max_model_tokens, std::nullopt, "Override max_model_tokens.");
ABSL_FLAG(std::optional<uint64_t>, max_output_tokens, std::nullopt, "Override max_output_tokens.");
ABSL_FLAG(std::optional<uint64_t>, max_prompt_tokens, std::nullopt, "Override max_prompt_tokens.");
ABSL_FLAG(std::optional<uint64_t>, max_queued_requests, std::nullopt,
          "Override max_queued_requests.");
ABSL_FLAG(std::optional<uint64_t>, max_scheduled_tokens_per_step, std::nullopt,
          "Override max_scheduled_tokens_per_step.");
ABSL_FLAG(std::optional<uint64_t>, max_sequences_per_step, std::nullopt,
          "Override max_sequences_per_step.");
ABSL_FLAG(std::optional<uint64_t>, max_simulation_events, std::nullopt,
          "Override max_simulation_events.");
ABSL_FLAG(std::optional<uint64_t>, plan_buffer_slots, std::nullopt, "Override plan_buffer_slots.");
ABSL_FLAG(std::optional<uint64_t>, response_channel_capacity, std::nullopt,
          "Override response_channel_capacity.");
ABSL_FLAG(std::optional<uint64_t>, simulated_kv_token_capacity, std::nullopt,
          "Override simulated_kv_token_capacity.");
ABSL_FLAG(std::optional<uint64_t>, submission_channel_capacity, std::nullopt,
          "Override submission_channel_capacity.");
ABSL_FLAG(std::optional<uint64_t>, fake_base_latency_ns, std::nullopt,
          "Override fake_base_latency_ns.");
ABSL_FLAG(std::optional<uint64_t>, fake_prefill_latency_per_token_ns, std::nullopt,
          "Override fake_prefill_latency_per_token_ns.");
ABSL_FLAG(std::optional<uint64_t>, fake_decode_latency_per_sequence_ns, std::nullopt,
          "Override fake_decode_latency_per_sequence_ns.");

namespace {

constexpr int kSuccess = 0;
constexpr int kValidationFailure = 2;
constexpr int kSimulationFailure = 3;
constexpr int kCorrectnessFailure = 4;
constexpr int kIoFailure = 5;

enum class FlagValueKind : uint8_t { kPath, kUnsigned, kBoolean };

std::optional<FlagValueKind> FlagKind(absl::string_view name) {
  if (name == "config" || name == "workload" || name == "trace" || name == "output") {
    return FlagValueKind::kPath;
  }
  if (name == "overwrite" || name == "verbose") return FlagValueKind::kBoolean;
#define INFERX_MATCH_SCHEMA_FLAG(camel, json_name, default_value) \
  if (name == (json_name)) return FlagValueKind::kUnsigned;
  INFERX_CONFIG_FIELDS(INFERX_MATCH_SCHEMA_FLAG)
#undef INFERX_MATCH_SCHEMA_FLAG
  return std::nullopt;
}

absl::Status PrevalidateArguments(int argc, char** argv) {
  for (int index = 1; index < argc; ++index) {
    const absl::string_view argument(argv[index]);
    if (!argument.starts_with("--")) continue;
    const size_t equals = argument.find('=');
    const absl::string_view name = argument.substr(
        2, equals == absl::string_view::npos ? absl::string_view::npos : equals - 2);
    if (name == "help" || name == "helpshort" || name == "helpfull") continue;
    const std::optional<FlagValueKind> kind = FlagKind(name);
    if (!kind.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat("cli.flags: unknown flag --", name));
    }
    absl::string_view value;
    if (equals != absl::string_view::npos) {
      value = argument.substr(equals + 1);
    } else if (*kind == FlagValueKind::kBoolean) {
      continue;
    } else {
      if (index + 1 >= argc || absl::string_view(argv[index + 1]).starts_with("--")) {
        return absl::InvalidArgumentError(absl::StrCat("cli.flags: --", name, " requires a value"));
      }
      value = argv[++index];
    }
    if (*kind == FlagValueKind::kUnsigned) {
      absl::StatusOr<uint64_t> parsed = inferx::config::ParseConfigInteger(value, name);
      if (!parsed.ok()) return parsed.status();
    } else if (*kind == FlagValueKind::kBoolean && value != "true" && value != "false" &&
               value != "1" && value != "0") {
      return absl::InvalidArgumentError(
          absl::StrCat("cli.flags: --", name, " must be true or false"));
    } else if (*kind == FlagValueKind::kPath && value.empty()) {
      return absl::InvalidArgumentError(absl::StrCat("cli.flags: --", name, " is empty"));
    }
  }
  return absl::OkStatus();
}

inferx::config::ModelCapabilities FakeModel() {
  return {.max_context_tokens = inferx::TokenCount(32768), .accepts_text = false};
}

inferx::config::FieldValues CommandLineValues() {
  inferx::config::FieldValues values;
#define INFERX_COPY_FLAG(flag_name)                                 \
  if (const auto value = absl::GetFlag(FLAGS_##flag_name); value) { \
    values.emplace(#flag_name, *value);                             \
  }
  INFERX_COPY_FLAG(max_active_sequences)
  INFERX_COPY_FLAG(max_model_tokens)
  INFERX_COPY_FLAG(max_output_tokens)
  INFERX_COPY_FLAG(max_prompt_tokens)
  INFERX_COPY_FLAG(max_queued_requests)
  INFERX_COPY_FLAG(max_scheduled_tokens_per_step)
  INFERX_COPY_FLAG(max_sequences_per_step)
  INFERX_COPY_FLAG(max_simulation_events)
  INFERX_COPY_FLAG(plan_buffer_slots)
  INFERX_COPY_FLAG(response_channel_capacity)
  INFERX_COPY_FLAG(simulated_kv_token_capacity)
  INFERX_COPY_FLAG(submission_channel_capacity)
  INFERX_COPY_FLAG(fake_base_latency_ns)
  INFERX_COPY_FLAG(fake_prefill_latency_per_token_ns)
  INFERX_COPY_FLAG(fake_decode_latency_per_sequence_ns)
#undef INFERX_COPY_FLAG
  return values;
}

absl::StatusOr<inferx::config::ParsedConfig> ReadParsedConfig(bool require_file) {
  const std::string path = absl::GetFlag(FLAGS_config);
  if (require_file && path.empty()) {
    return absl::InvalidArgumentError("cli.config: --config is required");
  }
  std::optional<inferx::config::FieldValues> file_values;
  if (!path.empty()) {
    absl::StatusOr<inferx::config::FieldValues> loaded = inferx::config::ReadConfigFile(path);
    if (!loaded.ok()) return loaded.status();
    file_values = std::move(*loaded);
  }
  absl::StatusOr<inferx::config::FieldValues> environment = inferx::config::ReadConfigEnvironment();
  if (!environment.ok()) return environment.status();
  return inferx::config::LoadConfig(file_values, std::move(*environment), CommandLineValues());
}

absl::StatusOr<inferx::config::EngineConfig> ReadEffectiveConfig(bool require_file) {
  absl::StatusOr<inferx::config::ParsedConfig> parsed = ReadParsedConfig(require_file);
  if (!parsed.ok()) return parsed.status();
  return inferx::config::EngineConfig::Validate(*parsed, inferx::config::BuildCapabilities{},
                                                FakeModel());
}

absl::string_view SourceName(inferx::config::ConfigSource source) {
  switch (source) {
    case inferx::config::ConfigSource::kDefault:
      return "default";
    case inferx::config::ConfigSource::kFile:
      return "file";
    case inferx::config::ConfigSource::kEnvironment:
      return "environment";
    case inferx::config::ConfigSource::kCommandLine:
      return "command-line";
  }
  return "unknown";
}

void PrintExplanation(const inferx::config::ParsedConfig& parsed) {
  std::vector<std::pair<std::string, inferx::config::SourcedValue>> values;
#define INFERX_EXPLAIN(camel, json_name, default_value) \
  values.emplace_back(json_name, parsed.camel);
  INFERX_CONFIG_FIELDS(INFERX_EXPLAIN)
#undef INFERX_EXPLAIN
  std::sort(values.begin(), values.end(),
            [](const auto& left, const auto& right) { return left.first < right.first; });
  for (const auto& [name, value] : values) {
    std::printf("%s=%llu source=%.*s\n", name.c_str(), static_cast<unsigned long long>(value.value),
                static_cast<int>(SourceName(value.source).size()), SourceName(value.source).data());
  }
}

int ExitForStatus(const absl::Status& status, int fallback) {
  if (status.ok()) return kSuccess;
  const auto reason = inferx::GetErrorReason(status);
  if (reason.ok()) {
    if (*reason == inferx::ErrorReason::kIoFailure) return kIoFailure;
    if (*reason == inferx::ErrorReason::kInvariantViolation ||
        *reason == inferx::ErrorReason::kReplayMismatch) {
      return kCorrectnessFailure;
    }
  }
  return fallback;
}

int Fail(const absl::Status& status, int fallback) {
  std::fprintf(stderr, "inferx-sim: %s\n", status.ToString().c_str());
  return ExitForStatus(status, fallback);
}

int ValidateConfig() {
  absl::StatusOr<inferx::config::EngineConfig> config = ReadEffectiveConfig(true);
  if (!config.ok()) return Fail(config.status(), kValidationFailure);
  std::printf("valid\n");
  return kSuccess;
}

int ExplainConfig() {
  absl::StatusOr<inferx::config::ParsedConfig> parsed = ReadParsedConfig(true);
  if (!parsed.ok()) return Fail(parsed.status(), kValidationFailure);
  absl::StatusOr<inferx::config::EngineConfig> effective = inferx::config::EngineConfig::Validate(
      *parsed, inferx::config::BuildCapabilities{}, FakeModel());
  if (!effective.ok()) return Fail(effective.status(), kValidationFailure);
  PrintExplanation(*parsed);
  return kSuccess;
}

int RunSimulation(const inferx::config::EngineConfig& config,
                  const std::vector<inferx::simulator::WorkloadEvent>& workload,
                  const std::string& trace_path, bool overwrite) {
  if (trace_path.empty()) {
    return Fail(absl::InvalidArgumentError("cli.trace: output path is required"),
                kValidationFailure);
  }
  auto sink = inferx::simulator::FileReplaySink::Create(trace_path, overwrite);
  if (!sink.ok()) return Fail(sink.status(), kIoFailure);
  auto simulator = inferx::simulator::EngineSimulator::Create(config, FakeModel(), **sink);
  if (!simulator.ok()) return Fail(simulator.status(), kValidationFailure);
  absl::Status loaded = (*simulator)->LoadWorkload(workload);
  if (!loaded.ok()) return Fail(loaded, kValidationFailure);
  auto result = (*simulator)->Run();
  if (!result.ok()) return Fail(result.status(), kSimulationFailure);
  const auto& summary = result->summary;
  std::printf(
      "events=%llu steps=%llu requests=%llu finished=%llu cancelled=%llu failed=%llu "
      "resources=%llu/%llu tickets=%llu plans=%llu invariants=%s\n",
      static_cast<unsigned long long>(summary.events),
      static_cast<unsigned long long>(summary.steps),
      static_cast<unsigned long long>(summary.requests),
      static_cast<unsigned long long>(summary.finished),
      static_cast<unsigned long long>(summary.cancelled),
      static_cast<unsigned long long>(summary.failed),
      static_cast<unsigned long long>(summary.used_sequences),
      static_cast<unsigned long long>(summary.used_kv_tokens),
      static_cast<unsigned long long>(summary.live_tickets),
      static_cast<unsigned long long>(summary.leased_plan_slots),
      summary.invariants_ok ? "ok" : "failed");
  if (!result->status.ok()) return Fail(result->status, kSimulationFailure);
  return kSuccess;
}

int Run() {
  const std::string workload_path = absl::GetFlag(FLAGS_workload);
  if (workload_path.empty()) {
    return Fail(absl::InvalidArgumentError("cli.workload: --workload is required"),
                kValidationFailure);
  }
  auto config = ReadEffectiveConfig(true);
  if (!config.ok()) return Fail(config.status(), kValidationFailure);
  auto workload = inferx::simulator::ReadWorkloadFile(workload_path, true);
  if (!workload.ok()) return Fail(workload.status(), kValidationFailure);
  if (absl::GetFlag(FLAGS_verbose)) {
    std::printf("config=%s\n", config->CanonicalJson().c_str());
  }
  return RunSimulation(*config, *workload, absl::GetFlag(FLAGS_trace),
                       absl::GetFlag(FLAGS_overwrite));
}

int CheckTrace() {
  const std::string trace_path = absl::GetFlag(FLAGS_trace);
  if (trace_path.empty()) {
    return Fail(absl::InvalidArgumentError("cli.trace: --trace is required"), kValidationFailure);
  }
  auto replay = inferx::simulator::ReadReplayInputs(trace_path);
  if (!replay.ok()) return Fail(replay.status(), kCorrectnessFailure);
  std::printf("valid records trace=%s\n", trace_path.c_str());
  return kSuccess;
}

int Replay() {
  const std::string trace_path = absl::GetFlag(FLAGS_trace);
  const std::string output_path = absl::GetFlag(FLAGS_output);
  if (trace_path.empty() || output_path.empty()) {
    return Fail(absl::InvalidArgumentError("cli.replay: --trace and --output are required"),
                kValidationFailure);
  }
  auto inputs = inferx::simulator::ReadReplayInputs(trace_path);
  if (!inputs.ok()) return Fail(inputs.status(), kCorrectnessFailure);
  auto parsed = inferx::config::LoadConfig(inputs->config_values, std::nullopt, std::nullopt);
  if (!parsed.ok()) return Fail(parsed.status(), kCorrectnessFailure);
  auto config = inferx::config::EngineConfig::Validate(*parsed, inferx::config::BuildCapabilities{},
                                                       FakeModel());
  if (!config.ok()) return Fail(config.status(), kCorrectnessFailure);
  const int run_exit =
      RunSimulation(*config, inputs->workload, output_path, absl::GetFlag(FLAGS_overwrite));
  if (run_exit != kSuccess) return run_exit;
  auto replayed = inferx::simulator::ReadReplayInputs(output_path);
  if (!replayed.ok()) return Fail(replayed.status(), kCorrectnessFailure);
  const absl::Status compared =
      inferx::simulator::CompareReplayBytes(inputs->original_bytes, replayed->original_bytes);
  if (!compared.ok()) return Fail(compared, kCorrectnessFailure);
  std::printf("replay=byte-identical\n");
  return kSuccess;
}

void PrintUsage() {
  std::fprintf(stderr,
               "usage: inferx-sim <validate-config|explain-config|run|replay|check-trace> "
               "[explicit flags]\n");
}

int Main(int argc, char** argv) {
  absl::SetProgramUsageMessage(
      "Deterministic InferX M1 configuration, simulator, and replay tool.");
  const absl::Status arguments = PrevalidateArguments(argc, argv);
  if (!arguments.ok()) return Fail(arguments, kValidationFailure);
  absl::DisableFlagfileAndEnvParsing();
  const std::vector<char*> positional = absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  if (positional.size() != 2) {
    PrintUsage();
    return kValidationFailure;
  }
  const absl::string_view command(positional[1]);
  if (command == "validate-config") return ValidateConfig();
  if (command == "explain-config") return ExplainConfig();
  if (command == "run") return Run();
  if (command == "replay") return Replay();
  if (command == "check-trace") return CheckTrace();
  PrintUsage();
  return kValidationFailure;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Main(argc, argv);
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "inferx-sim: unexpected exception: %s\n", exception.what());
    return kCorrectnessFailure;
  } catch (...) {
    std::fprintf(stderr, "inferx-sim: unexpected non-standard exception\n");
    return kCorrectnessFailure;
  }
}
