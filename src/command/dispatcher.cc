#include "inferx/command/dispatcher.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cuda_environment.h"
#include "inferx/api/response_event.h"
#include "inferx/artifacts/artifact_tensor.h"
#include "inferx/artifacts/model_resolver.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"
#include "inferx/base/version.h"
#include "inferx/config/config_loader.h"
#include "inferx/config/engine_config.h"
#include "inferx/config/parsed_config.h"
#include "inferx/engine/event_sink.h"
#include "inferx/engine/request_event.h"
#include "inferx/engine/request_state_machine.h"
#include "inferx/input/model_package.h"
#include "inferx/input/prompt_input.h"
#include "inferx/input/prompt_processor.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/runtime/cpu_execution_backend.h"
#include "inferx/runtime/execution_backend.h"
#include "inferx/runtime/single_request_runner.h"
#include "inferx/simulator/engine_simulator.h"
#include "inferx/simulator/replay.h"
#include "inferx/simulator/workload.h"
#include "inferx/tokenization/tokenizer_options.h"

namespace inferx::command {
namespace {

artifacts::ModelResolverOptions ResolverOptions(const command::ResolverOptions& options) {
  artifacts::ModelResolverOptions result;
  result.revision = options.revision;
  result.local_files_only = options.offline;
  if (!options.download_dir.empty())
    result.download_dir = std::filesystem::path(options.download_dir);
  return result;
}

std::string JsonString(std::string_view value) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size() + 2);
  result.push_back('"');
  for (const char character : value) {
    const auto byte = static_cast<unsigned char>(character);
    switch (byte) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\b':
        result += "\\b";
        break;
      case '\f':
        result += "\\f";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (byte < 0x20) {
          result += "\\u00";
          result.push_back(kHex[byte >> 4]);
          result.push_back(kHex[byte & 0x0F]);
        } else {
          result.push_back(static_cast<char>(byte));
        }
    }
  }
  result.push_back('"');
  return result;
}

void PrintTokenIds(std::ostream& output, std::span<const TokenId> tokens) {
  output << '[';
  for (size_t index = 0; index < tokens.size(); ++index) {
    if (index != 0) output << ',';
    output << tokens[index].value();
  }
  output << ']';
}

void PrintFsmSchema(std::ostream& output) {
  const TransitionRuleSpan rules = TransitionRules();
  output << "{\"schema_version\":1,\"transitions\":[";
  for (size_t index = 0; index < rules.size; ++index) {
    const TransitionRule& rule = rules.data[index];
    if (index != 0) output << ',';
    output << "{\"from\":" << JsonString(ToString(rule.current))
           << ",\"event\":" << JsonString(ToString(rule.event)) << ",\"to\":";
    if (rule.resolve_next == nullptr) {
      output << JsonString(ToString(rule.fixed_next));
    } else {
      output << "null";
    }
    output << ",\"conditional\":" << (rule.resolve_next == nullptr ? "false" : "true")
           << ",\"effects\":" << static_cast<unsigned>(rule.effects)
           << ",\"terminal\":" << (rule.terminal ? "true" : "false") << '}';
  }
  output << "]}\n";
}

class DiscardingResponseSink final : public ResponseSink {
 public:
  absl::StatusOr<ResponseReservation> Prepare(ResponseEvent event) override {
    return ResponseReservation(std::move(event));
  }
  void Commit(ResponseReservation) noexcept override {}
};

config::ModelCapabilities FakeModel() {
  return {.max_context_tokens = TokenCount(32768), .accepts_text = false};
}

config::FieldValues CommandLineValues(const SimulateOptions& options) {
  config::FieldValues values;
#define INFERX_COPY_OVERRIDE(field)         \
  if (options.field.has_value()) {          \
    values.emplace(#field, *options.field); \
  }
  INFERX_COPY_OVERRIDE(max_active_sequences)
  INFERX_COPY_OVERRIDE(max_model_tokens)
  INFERX_COPY_OVERRIDE(max_output_tokens)
  INFERX_COPY_OVERRIDE(max_prompt_tokens)
  INFERX_COPY_OVERRIDE(max_queued_requests)
  INFERX_COPY_OVERRIDE(max_scheduled_tokens_per_step)
  INFERX_COPY_OVERRIDE(max_sequences_per_step)
  INFERX_COPY_OVERRIDE(max_simulation_events)
  INFERX_COPY_OVERRIDE(plan_buffer_slots)
  INFERX_COPY_OVERRIDE(response_channel_capacity)
  INFERX_COPY_OVERRIDE(simulated_kv_token_capacity)
  INFERX_COPY_OVERRIDE(submission_channel_capacity)
  INFERX_COPY_OVERRIDE(fake_base_latency_ns)
  INFERX_COPY_OVERRIDE(fake_prefill_latency_per_token_ns)
  INFERX_COPY_OVERRIDE(fake_decode_latency_per_sequence_ns)
#undef INFERX_COPY_OVERRIDE
  return values;
}

absl::StatusOr<config::ParsedConfig> ReadParsedConfig(const SimulateOptions& options) {
  std::optional<config::FieldValues> file_values;
  if (!options.config.empty()) {
    auto loaded = config::ReadConfigFile(options.config);
    if (!loaded.ok()) return loaded.status();
    file_values = std::move(*loaded);
  }
  auto environment = config::ReadConfigEnvironment();
  if (!environment.ok()) return environment.status();
  return config::LoadConfig(file_values, std::move(*environment), CommandLineValues(options));
}

absl::StatusOr<config::EngineConfig> ReadEffectiveConfig(const SimulateOptions& options) {
  auto parsed = ReadParsedConfig(options);
  if (!parsed.ok()) return parsed.status();
  return config::EngineConfig::Validate(*parsed, config::BuildCapabilities{}, FakeModel());
}

std::string_view SourceName(config::ConfigSource source) {
  switch (source) {
    case config::ConfigSource::kDefault:
      return "default";
    case config::ConfigSource::kFile:
      return "file";
    case config::ConfigSource::kEnvironment:
      return "environment";
    case config::ConfigSource::kCommandLine:
      return "command-line";
  }
  return "unknown";
}

void PrintExplanation(std::ostream& output, const config::ParsedConfig& parsed) {
  std::vector<std::pair<std::string, config::SourcedValue>> values;
#define INFERX_EXPLAIN(camel, json_name, default_value) \
  values.emplace_back(json_name, parsed.camel);
  INFERX_CONFIG_FIELDS(INFERX_EXPLAIN)
#undef INFERX_EXPLAIN
  std::sort(values.begin(), values.end(),
            [](const auto& left, const auto& right) { return left.first < right.first; });
  for (const auto& [name, value] : values) {
    output << name << '=' << value.value << " source=" << SourceName(value.source) << '\n';
  }
}

absl::Status RunSimulation(std::ostream& output, const config::EngineConfig& config,
                           const std::vector<simulator::WorkloadEvent>& workload,
                           const std::string& trace_path, bool overwrite) {
  auto sink = simulator::FileReplaySink::Create(trace_path, overwrite);
  if (!sink.ok()) return sink.status();
  auto engine = simulator::EngineSimulator::Create(config, FakeModel(), **sink);
  if (!engine.ok()) return engine.status();
  absl::Status loaded = (*engine)->LoadWorkload(workload);
  if (!loaded.ok()) return loaded;
  auto result = (*engine)->Run();
  if (!result.ok()) return result.status();
  const auto& summary = result->summary;
  output << "events=" << summary.events << " steps=" << summary.steps
         << " requests=" << summary.requests << " finished=" << summary.finished
         << " cancelled=" << summary.cancelled << " failed=" << summary.failed
         << " resources=" << summary.used_sequences << '/' << summary.used_kv_tokens
         << " tickets=" << summary.live_tickets << " plans=" << summary.leased_plan_slots
         << " invariants=" << (summary.invariants_ok ? "ok" : "failed") << '\n';
  return result->status;
}

class DefaultDispatcher final : public Dispatcher {
 public:
  DefaultDispatcher(std::ostream& output, std::ostream& error) : output_(output), error_(error) {}

  ExitCode Serve(const GlobalOptions&, const ServeOptions&) override {
    return Unavailable("serve", "HTTP serving is scheduled for the server milestone");
  }

  ExitCode Bench(const GlobalOptions&, const BenchmarkOptions&) override {
    return Unavailable("bench", "the unified benchmark runner is not compiled in this milestone");
  }

  ExitCode Run(const GlobalOptions& global, const RunOptions& options) override {
    if ((!options.model.tokenizer.empty()) ||
        (options.model.device != "auto" && options.model.device != "cpu") ||
        options.model.tensor_parallel_size != 1 ||
        (options.model.dtype != DType::kAuto && options.model.dtype != DType::kFloat32)) {
      return Unavailable(
          "run", "the current local runner supports the model tokenizer, one CPU, and float32");
    }
    if (options.sampling.temperature != 0.0 || options.sampling.top_p != 1.0 ||
        options.sampling.top_k != 0) {
      error_ << "error: run: the current backend supports deterministic greedy decoding only\n";
      return ExitCode::kUsage;
    }

    artifacts::ModelResolver resolver;
    auto resolved = resolver.Resolve(options.model.model, ResolverOptions(options.resolver));
    if (!resolved.ok()) return Fail("model resolution failed", resolved.status(), ExitCode::kModel);
    auto package = input::ModelPackageLoader::Load(resolved->path.string());
    if (!package.ok()) return Fail("model loading failed", package.status(), ExitCode::kModel);
    const model::LlamaSpec& llama = package->model_spec().llama();
    if (llama.max_position_embeddings > (std::numeric_limits<uint32_t>::max)()) {
      return Fail("model loading failed",
                  absl::OutOfRangeError("model context exceeds the request token-count range"),
                  ExitCode::kModel);
    }
    uint64_t context_capacity = llama.max_position_embeddings;
    if (options.model.max_model_len != 0) {
      context_capacity = (std::min)(context_capacity, options.model.max_model_len);
    }
    if (context_capacity > (std::numeric_limits<uint32_t>::max)()) {
      return Fail("configuration failed", absl::OutOfRangeError("max model length is too large"),
                  ExitCode::kUsage);
    }

    input::InputProcessingRequest input_request{RequestId(1),
                                                RequestEpoch(1),
                                                ModelId(0),
                                                input::RawTextPrompt{options.prompt, true},
                                                TokenCount(options.max_tokens),
                                                std::nullopt};
    const input::PromptModelFacts facts{&package->tokenizer(), llama.vocab_size,
                                        TokenCount(static_cast<uint32_t>(context_capacity))};
    auto request = input::PromptProcessor::Process(input_request, facts, input::PromptLimits{},
                                                   MonotonicTime{});
    if (!request.ok()) return Fail("request validation failed", request.status(), ExitCode::kUsage);
    const auto* prompt_tokens = std::get_if<std::vector<TokenId>>(&request->input);
    if (prompt_tokens == nullptr) {
      return Fail("inference failed", absl::InternalError("prompt processor produced no token IDs"),
                  ExitCode::kRuntime);
    }
    const std::vector<TokenId> prompt_token_ids = *prompt_tokens;

    runtime::CpuExecutionBackend backend;
    runtime::ModelLoadPlan load_plan;
    load_plan.package = &*package;
    load_plan.model_root = resolved->path.string();
    load_plan.max_prefill_tokens = prompt_token_ids.size();
    load_plan.context_capacity = context_capacity;
    auto handle = backend.Load(load_plan);
    if (!handle.ok())
      return Fail("backend model loading failed", handle.status(), ExitCode::kRuntime);

    DiscardingResponseSink sink;
    auto runner = runtime::SingleRequestRunner::Create(backend, *handle, load_plan.context_capacity,
                                                       load_plan.max_prefill_tokens, sink);
    if (!runner.ok()) return Fail("inference setup failed", runner.status(), ExitCode::kRuntime);
    auto generated = (*runner)->Run(std::move(*request), MonotonicTime{});
    if (!generated.ok()) return Fail("inference failed", generated.status(), ExitCode::kRuntime);
    if (!generated->status.ok()) {
      return Fail("inference failed", generated->status, ExitCode::kRuntime);
    }

    tokenization::DecodeOptions decode_options;
    decode_options.skip_special_tokens = true;
    auto decoded = package->tokenizer().Decode(generated->output_tokens, decode_options);
    if (!decoded.ok()) return Fail("decode failed", decoded.status(), ExitCode::kRuntime);

    output_ << "{\"model\":" << JsonString(resolved->model)
            << ",\"model_path\":" << JsonString(resolved->path.string())
            << ",\"model_revision\":" << JsonString(resolved->revision)
            << ",\"seed\":" << global.seed << ",\"temperature\":" << options.sampling.temperature
            << ",\"prompt_token_ids\":";
    PrintTokenIds(output_, prompt_token_ids);
    output_ << ",\"output_token_ids\":";
    PrintTokenIds(output_, generated->output_tokens);
    output_ << ",\"output_text\":" << JsonString(*decoded)
            << ",\"finish_reason\":" << JsonString(ToString(generated->finish)) << "}\n";

    runner->reset();
    const absl::Status unloaded = backend.Unload(*handle);
    if (!unloaded.ok()) return Fail("backend unload failed", unloaded, ExitCode::kRuntime);
    return ExitCode::kSuccess;
  }

  ExitCode Chat(const GlobalOptions&, const ClientOptions&) override {
    return Unavailable("chat",
                       "the OpenAI-compatible HTTP client is not compiled in this milestone");
  }

  ExitCode Complete(const GlobalOptions&, const ClientOptions&) override {
    return Unavailable("complete",
                       "the OpenAI-compatible HTTP client is not compiled in this milestone");
  }

  ExitCode Inspect(const GlobalOptions&, const InspectOptions& options) override {
    if (options.fsm_schema) {
      PrintFsmSchema(output_);
      return ExitCode::kSuccess;
    }
    artifacts::ModelResolver resolver;
    auto resolved = resolver.Resolve(options.model.model, ResolverOptions(options.resolver));
    if (!resolved.ok()) return Fail("model resolution failed", resolved.status(), ExitCode::kModel);
    input::PackageTokenizerPolicy policy;
    auto package = input::ModelPackageLoader::Load(resolved->path.string(), policy);
    if (!package.ok()) return Fail("model inspection failed", package.status(), ExitCode::kModel);
    const auto& llama = package->model_spec().llama();
    const auto& tokenizer = package->tokenizer_metadata();
    uint64_t estimated_weight_bytes = 0;
    bool size_overflow = false;
    for (const auto& item : package->weight_plan().items) {
      if (item.source.file_range.size >
          (std::numeric_limits<uint64_t>::max)() - estimated_weight_bytes) {
        size_overflow = true;
        break;
      }
      estimated_weight_bytes += item.source.file_range.size;
    }
    output_ << "model=" << resolved->model << '\n'
            << "model_source=" << artifacts::ModelSourceName(resolved->source) << '\n'
            << "model_path=" << resolved->path.string() << '\n'
            << "model_revision=" << resolved->revision << '\n'
            << "architecture=llama\n"
            << "weights_entry=" << package->weights_entry().string() << '\n'
            << "dtype="
            << (llama.source_weight_type_hint.has_value()
                    ? artifacts::ArtifactDtypeName(*llama.source_weight_type_hint)
                    : std::string_view("mixed-or-unspecified"))
            << '\n'
            << "quantization=none-or-unreported\n"
            << "context_length=" << llama.max_position_embeddings << '\n'
            << "vocab_size=" << llama.vocab_size << '\n'
            << "hidden_size=" << llama.hidden_size << '\n'
            << "intermediate_size=" << llama.intermediate_size << '\n'
            << "layers=" << llama.num_hidden_layers << '\n'
            << "attention_heads=" << llama.num_attention_heads << '\n'
            << "key_value_heads=" << llama.num_key_value_heads << '\n'
            << "head_dim=" << llama.head_dim << '\n'
            << "tensor_count=" << package->weight_plan().items.size() << '\n'
            << "estimated_weight_bytes=";
    if (size_overflow) {
      output_ << "overflow\n";
    } else {
      output_ << estimated_weight_bytes << '\n';
    }
    output_ << "tokenizer_backend=" << tokenizer.backend << '\n'
            << "tokenizer_engine_version=" << tokenizer.engine_version << '\n'
            << "tokenizer_vocab_size=" << tokenizer.total_vocab_size << '\n'
            << "parameters_expected=" << package->weight_plan().coverage.expected << '\n'
            << "parameters_assigned=" << package->weight_plan().coverage.assigned << '\n'
            << "parameters_aliased=" << package->weight_plan().coverage.aliased << '\n'
            << "integrity_manifest=" << (package->integrity_manifest() ? "true" : "false") << '\n'
            << "tokenizer_qualified=true\n"
            << "model_fingerprint=" << package->fingerprint().digest().Hex() << '\n';
    for (const auto& parameter : package->parameters()) {
      output_ << "tensor." << parameter.canonical_name << "=[";
      for (size_t index = 0; index < parameter.shape.size(); ++index) {
        if (index != 0) output_ << ',';
        output_ << parameter.shape[index];
      }
      output_ << "]\n";
    }
    return ExitCode::kSuccess;
  }

  ExitCode Download(const GlobalOptions&, const DownloadOptions& options) override {
    artifacts::ModelResolver resolver;
    auto resolver_options = ResolverOptions(options.resolver);
    auto resolved = resolver.Resolve(options.model, resolver_options);
    if (!resolved.ok()) return Fail("model download failed", resolved.status(), ExitCode::kModel);
    output_ << "model=" << resolved->model << '\n'
            << "model_source=" << artifacts::ModelSourceName(resolved->source) << '\n'
            << "model_path=" << resolved->path.string() << '\n'
            << "model_revision=" << resolved->revision << '\n';
    return ExitCode::kSuccess;
  }

  ExitCode Simulate(const GlobalOptions&, const SimulateOptions& options) override {
    if (options.operation == SimulateOperation::kValidateConfig) {
      auto config = ReadEffectiveConfig(options);
      if (!config.ok())
        return Fail("simulate validate-config failed", config.status(), ExitCode::kUsage);
      output_ << "valid\n";
      return ExitCode::kSuccess;
    }
    if (options.operation == SimulateOperation::kExplainConfig) {
      auto parsed = ReadParsedConfig(options);
      if (!parsed.ok())
        return Fail("simulate explain-config failed", parsed.status(), ExitCode::kUsage);
      auto effective =
          config::EngineConfig::Validate(*parsed, config::BuildCapabilities{}, FakeModel());
      if (!effective.ok()) {
        return Fail("simulate explain-config failed", effective.status(), ExitCode::kUsage);
      }
      PrintExplanation(output_, *parsed);
      return ExitCode::kSuccess;
    }
    if (options.operation == SimulateOperation::kCheckTrace) {
      auto replay = simulator::ReadReplayInputs(options.trace);
      if (!replay.ok())
        return Fail("simulate check-trace failed", replay.status(), ExitCode::kRuntime);
      output_ << "valid records trace=" << options.trace << '\n';
      return ExitCode::kSuccess;
    }
    if (options.operation == SimulateOperation::kRun) {
      auto config = ReadEffectiveConfig(options);
      if (!config.ok()) return Fail("simulate run failed", config.status(), ExitCode::kUsage);
      auto workload = simulator::ReadWorkloadFile(options.workload, true);
      if (!workload.ok()) return Fail("simulate run failed", workload.status(), ExitCode::kUsage);
      if (options.verbose) output_ << "config=" << config->CanonicalJson() << '\n';
      const absl::Status status =
          RunSimulation(output_, *config, *workload, options.trace, options.overwrite);
      if (!status.ok()) return Fail("simulate run failed", status, ExitCode::kRuntime);
      return ExitCode::kSuccess;
    }

    auto inputs = simulator::ReadReplayInputs(options.trace);
    if (!inputs.ok()) return Fail("simulate replay failed", inputs.status(), ExitCode::kRuntime);
    auto parsed = config::LoadConfig(inputs->config_values, std::nullopt, std::nullopt);
    if (!parsed.ok()) return Fail("simulate replay failed", parsed.status(), ExitCode::kRuntime);
    auto effective =
        config::EngineConfig::Validate(*parsed, config::BuildCapabilities{}, FakeModel());
    if (!effective.ok())
      return Fail("simulate replay failed", effective.status(), ExitCode::kRuntime);
    absl::Status status =
        RunSimulation(output_, *effective, inputs->workload, options.output, options.overwrite);
    if (!status.ok()) return Fail("simulate replay failed", status, ExitCode::kRuntime);
    auto replayed = simulator::ReadReplayInputs(options.output);
    if (!replayed.ok())
      return Fail("simulate replay failed", replayed.status(), ExitCode::kRuntime);
    status = simulator::CompareReplayBytes(inputs->original_bytes, replayed->original_bytes);
    if (!status.ok()) return Fail("simulate replay failed", status, ExitCode::kRuntime);
    output_ << "replay=byte-identical\n";
    return ExitCode::kSuccess;
  }

  ExitCode Version(const GlobalOptions&) override {
    const inferx::Version version = GetVersion();
    output_ << "inferx: " << version.major << '.' << version.minor << '.' << version.patch << '\n'
            << "git-revision: " << INFERX_COMMAND_GIT_REVISION << '\n'
            << "compiler: " << INFERX_COMMAND_CXX_COMPILER_ID << ' '
            << INFERX_COMMAND_CXX_COMPILER_VERSION << '\n'
            << "c++-standard: " << INFERX_COMMAND_CXX_STANDARD << '\n'
            << "build-type: " << INFERX_COMMAND_BUILD_TYPE << '\n'
            << "features: " << INFERX_COMMAND_FEATURES << '\n';
#if INFERX_COMMAND_ENABLE_CUDA
    output_ << "cuda-architectures: " << INFERX_COMMAND_CUDA_ARCHITECTURES << '\n';
#endif
    return ExitCode::kSuccess;
  }

  ExitCode Environment(const GlobalOptions&) override {
    output_ << "compiled.cuda=" << (INFERX_COMMAND_ENABLE_CUDA ? "true" : "false") << '\n'
            << "compiled.rocm=false\n"
            << "rocm.driver_version=not-compiled\n"
            << "rocm.runtime_version=not-compiled\n"
            << "rocm.device_count=0\n";
    internal::PrintCudaEnvironment(output_);
    PrintEnvironment("CUDA_VISIBLE_DEVICES");
    PrintEnvironment("HIP_VISIBLE_DEVICES");
    PrintEnvironment("ROCR_VISIBLE_DEVICES");
    return ExitCode::kSuccess;
  }

 private:
  ExitCode Unavailable(std::string_view command, std::string_view detail) {
    error_ << "error: " << command << ": feature unavailable: " << detail << '\n';
    return ExitCode::kUnavailable;
  }

  ExitCode Fail(std::string_view scope, const absl::Status& status, ExitCode code) {
    error_ << "error: " << scope << ": " << status << '\n';
    return code;
  }

  void PrintEnvironment(const char* name) {
    const char* value = std::getenv(name);
    output_ << "environment." << name << '=' << (value == nullptr ? "<unset>" : value) << '\n';
  }

  std::ostream& output_;
  std::ostream& error_;
};

}  // namespace

std::unique_ptr<Dispatcher> CreateDefaultDispatcher(std::ostream& output, std::ostream& error) {
  return std::make_unique<DefaultDispatcher>(output, error);
}

}  // namespace inferx::command
