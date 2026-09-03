#include "inferx/cli/app.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "CLI/CLI.hpp"
#include "absl/status/status.h"
#include "inferx/base/log.h"
#include "inferx/base/version.h"
#include "inferx/command/options.h"

namespace inferx::cli {
namespace {

using command::BenchmarkMode;
using command::DType;
using command::LogLevel;
using command::OutputFormat;

// Terminal-exit contract shared with main: parse and usage errors carry
// kUnknown, whose numeric value (2) is the process exit status
// (command::ExitCode::kUsage); help and --version exits carry kCancelled
// because their output is already printed and the process must exit 0
// without dispatching.
constexpr absl::StatusCode kUsageExit = absl::StatusCode::kUnknown;
constexpr absl::StatusCode kTerminalOutputExit = absl::StatusCode::kCancelled;

template <typename Enum>
CLI::Validator StrictEnumTransformer(std::map<std::string, Enum> values) {
  std::string description = "{";
  for (auto iterator = values.begin(); iterator != values.end(); ++iterator) {
    if (iterator != values.begin()) description += ',';
    description += iterator->first;
  }
  description += '}';
  return CLI::Validator(
      [values = std::move(values), description](std::string& value) {
        const auto iterator = values.find(CLI::ignore_case(value));
        if (iterator == values.end()) return std::string("expected one of ") + description;
        value = std::to_string(static_cast<unsigned int>(iterator->second));
        return std::string{};
      },
      description);
}

CLI::Validator DeviceValidator() {
  return CLI::Validator(
      [](std::string& value) {
        if (value == "auto" || value == "cpu" || value == "cuda" || value == "rocm") {
          return std::string{};
        }
        constexpr std::string_view kCudaPrefix = "cuda:";
        constexpr std::string_view kRocmPrefix = "rocm:";
        const std::string_view view(value);
        const std::string_view prefix = view.starts_with(kCudaPrefix) ? kCudaPrefix : kRocmPrefix;
        if (!view.starts_with(prefix) || view.size() == prefix.size()) {
          return std::string("expected auto, cpu, cuda, cuda:N, rocm, or rocm:N");
        }
        for (const char character : view.substr(prefix.size())) {
          if (character < '0' || character > '9') {
            return std::string("device index must be a non-negative integer");
          }
        }
        return std::string{};
      },
      "DEVICE");
}

CLI::Validator NonEmptyValidator(std::string name) {
  return CLI::Validator(
      [name = std::move(name)](std::string& value) {
        return value.empty() ? name + " must not be empty" : std::string{};
      },
      "NONEMPTY");
}

void AddModelOptions(CLI::App& app, command::ModelOptions& options, bool require_model) {
  CLI::Option* model =
      app.add_option("--model", options.model, "Hugging Face model ID or local model directory");
  model->check(NonEmptyValidator("model"));
  if (require_model) model->required();
  app.add_option("--tokenizer", options.tokenizer,
                 "Tokenizer ID or path (defaults to the model tokenizer)")
      ->check(NonEmptyValidator("tokenizer"));
  app.add_option("--device", options.device, "Execution device")
      ->check(DeviceValidator())
      ->capture_default_str();
  const std::map<std::string, DType> dtypes{{"auto", DType::kAuto},
                                            {"float32", DType::kFloat32},
                                            {"float16", DType::kFloat16},
                                            {"bfloat16", DType::kBFloat16}};
  app.add_option("--dtype", options.dtype, "Model/runtime data type")
      ->transform(StrictEnumTransformer(dtypes))
      ->default_str("auto");
  app.add_option("--tensor-parallel-size", options.tensor_parallel_size,
                 "Number of tensor-parallel devices")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();
  app.add_option("--max-model-len", options.max_model_len,
                 "Maximum model context length in tokens (0 uses the model limit)")
      ->capture_default_str();
  app.add_option("--max-batch-size", options.max_batch_size, "Maximum requests in one batch")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();
  // NOTE: ModelOptions also carries quantization, kv_cache_dtype,
  // load_format, trust_remote_code, served_model_name,
  // pipeline_parallel_size, distributed_executor_backend,
  // enforce_eager, swap_space_gib, and num_gpu_blocks_override, which
  // are not yet exposed as flags here. Add them when those knobs need
  // to be CLI-reachable rather than config-file-only.
}

void AddSamplingOptions(CLI::App& app, command::SamplingOptions& options) {
  app.add_option("--temperature", options.temperature,
                 "Sampling temperature; 0 selects deterministic greedy decoding")
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();
  // Lower bound excludes 0 so this matches the documented (0, 1] range and
  // CLI11 rejects an invalid value at parse time with a flag-specific
  // error, instead of a generic post-parse UsageError.
  app.add_option("--top-p", options.top_p, "Nucleus sampling probability in (0, 1]")
      ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0))
      ->capture_default_str();
  app.add_option("--top-k", options.top_k, "Top-k sampling limit; 0 disables it")
      ->capture_default_str();
}

void AddResolverOptions(CLI::App& app, command::ResolverOptions& options) {
  app.add_option("--revision", options.revision, "Hugging Face revision")
      ->check(NonEmptyValidator("revision"))
      ->capture_default_str();
  app.add_option("--download-dir", options.download_dir, "Model cache directory");
  app.add_flag("--offline", options.offline, "Use local files only");
}

void AddOutputFormat(CLI::App& app, OutputFormat& output_format) {
  const std::map<std::string, OutputFormat> formats{{"text", OutputFormat::kText},
                                                    {"json", OutputFormat::kJson}};
  app.add_option("--output-format", output_format, "Output format")
      ->transform(StrictEnumTransformer(formats))
      ->default_str("text");
}

bool ValidProbability(double value, bool allow_zero) {
  return std::isfinite(value) && value <= 1.0 && (allow_zero ? value >= 0.0 : value > 0.0);
}

bool ValidSampling(const command::SamplingOptions& options) {
  return std::isfinite(options.temperature) && options.temperature >= 0.0 &&
         ValidProbability(options.top_p, false);
}

// Prints the error once, here, and returns a Status carrying the same
// message for the process exit code. Callers should surface the returned
// Status's exit code but must not print status.message() again, or the
// user sees the error twice.
absl::Status UsageError(std::string_view command, std::string_view message) {
  std::cerr << "error: " << command << ": " << message << '\n';
  return absl::Status(kUsageExit, std::string(command) + ": " + std::string(message));
}

/// Binds one CLI11 subcommand to the typed options it populates.
/// `app` is non-null once `ParseFromCommandLine` has registered the
/// subcommand; check `app->parsed()` to see if the user selected it.
template <typename OptionsT>
struct SubcommandBinding {
  CLI::App* app = nullptr;
  OptionsT options;
};

/// The CLI11 application plus the option storage it binds. One instance
/// per `ParseFromCommandLine` call. `bench_latency`/`bench_throughput`/
/// `bench_serve` and `client_chat`/`client_complete` intentionally share
/// nothing extra beyond `app` — which one ran is read off
/// `BenchmarkOptions::mode` / `ClientOptions::interactive`, set at
/// registration time, not off which struct field got written.
struct CommandLine {
  CLI::App app{"InferX unified inference runtime", "inferx"};

  command::GlobalOptions global;

  SubcommandBinding<command::ServeOptions> serve;
  SubcommandBinding<command::BenchmarkOptions> bench_latency;
  SubcommandBinding<command::BenchmarkOptions> bench_throughput;
  SubcommandBinding<command::BenchmarkOptions> bench_serve;
  SubcommandBinding<command::RunOptions> run;
  SubcommandBinding<command::ClientOptions> client_chat;
  SubcommandBinding<command::ClientOptions> client_complete;
  SubcommandBinding<command::InspectOptions> inspect;
  SubcommandBinding<command::DownloadOptions> download;
  SubcommandBinding<command::VersionOptions> version;
  SubcommandBinding<command::CollectEnvOptions> collect_env;
};

void AddGlobalOptions(CommandLine& cli) {
  cli.app.set_help_flag("-h,--help", "Show help and exit");
  cli.app.set_version_flag("--version", std::string("inferx ") + std::string(GetVersionString()),
                           "Print the concise InferX version");
  cli.app.failure_message([](const CLI::App* failed_app, const CLI::Error& failure) {
    return "error: " + failed_app->get_display_name() + ": " + failure.what() +
           "\nRun with --help for more information.\n";
  });
  cli.app.positionals_at_end();
  cli.app.require_subcommand(1);

  const std::map<std::string, LogLevel> log_levels{{"trace", LogLevel::kTrace},
                                                   {"debug", LogLevel::kDebug},
                                                   {"info", LogLevel::kInfo},
                                                   {"warning", LogLevel::kWarning},
                                                   {"error", LogLevel::kError}};
  cli.app.add_option("--log-level", cli.global.log_level, "Logging severity")
      ->transform(StrictEnumTransformer(log_levels))
      ->default_str("info");
  cli.app.add_option("--log-file", cli.global.log_file, "Write logs to this file");
  cli.app.add_option("--seed", cli.global.seed, "Deterministic unsigned 64-bit seed")
      ->capture_default_str();
}

void AddServeCommand(CommandLine& cli) {
  cli.serve.app = cli.app.add_subcommand("serve", "Start an inference server");
  cli.serve.app->fallthrough();
  AddModelOptions(*cli.serve.app, cli.serve.options.model, true);
  AddSamplingOptions(*cli.serve.app, cli.serve.options.sampling);
  cli.serve.app->add_option("--host", cli.serve.options.host, "Listener host or address")
      ->check(NonEmptyValidator("host"))
      ->capture_default_str();
  cli.serve.app->add_option("--port", cli.serve.options.port, "Listener port")
      ->check(CLI::Range(uint16_t{1}, (std::numeric_limits<uint16_t>::max)()))
      ->capture_default_str();
  cli.serve.app
      ->add_option("--max-running-requests", cli.serve.options.max_running_requests,
                   "Maximum concurrently running requests")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();
  cli.serve.app
      ->add_option("--gpu-memory-utilization", cli.serve.options.gpu_memory_utilization,
                   "Fraction of GPU memory available to InferX")
      ->check(CLI::Range(0.0, 1.0))
      ->capture_default_str();
  // NOTE: ServeOptions also carries speculative decoding, LoRA, auth,
  // CORS, TLS, guided-decoding/tool-calling, and deployment fields not
  // yet exposed here. See the ModelOptions NOTE above for the same
  // caveat.
}

void AddBenchCommands(CommandLine& cli) {
  CLI::App* bench = cli.app.add_subcommand("bench", "Run inference performance benchmarks");
  bench->require_subcommand(1);
  bench->fallthrough();

  cli.bench_latency.app = bench->add_subcommand("latency", "Measure latency, TTFT, and TPOT");
  cli.bench_latency.app->fallthrough();
  cli.bench_latency.options.mode = BenchmarkMode::kLatency;
  AddModelOptions(*cli.bench_latency.app, cli.bench_latency.options.model, true);
  AddSamplingOptions(*cli.bench_latency.app, cli.bench_latency.options.sampling);
  AddOutputFormat(*cli.bench_latency.app, cli.bench_latency.options.output_format);
  cli.bench_latency.app
      ->add_option("--num-prompts", cli.bench_latency.options.num_prompts, "Measured request count")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();

  cli.bench_throughput.app =
      bench->add_subcommand("throughput", "Measure request and token throughput");
  cli.bench_throughput.app->fallthrough();
  cli.bench_throughput.options.mode = BenchmarkMode::kThroughput;
  AddModelOptions(*cli.bench_throughput.app, cli.bench_throughput.options.model, true);
  AddSamplingOptions(*cli.bench_throughput.app, cli.bench_throughput.options.sampling);
  AddOutputFormat(*cli.bench_throughput.app, cli.bench_throughput.options.output_format);
  cli.bench_throughput.app
      ->add_option("--num-prompts", cli.bench_throughput.options.num_prompts,
                   "Measured request count")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();

  cli.bench_serve.app = bench->add_subcommand("serve", "Benchmark a running inferx serve instance");
  cli.bench_serve.app->fallthrough();
  cli.bench_serve.options.mode = BenchmarkMode::kServe;
  cli.bench_serve.app->add_option("--endpoint", cli.bench_serve.options.endpoint, "Server base URL")
      ->check(NonEmptyValidator("endpoint"))
      ->capture_default_str();
  cli.bench_serve.app->add_option("--model", cli.bench_serve.options.model.model,
                                  "Served model name");
  AddOutputFormat(*cli.bench_serve.app, cli.bench_serve.options.output_format);
  cli.bench_serve.app
      ->add_option("--num-prompts", cli.bench_serve.options.num_prompts, "Measured request count")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();
}

void AddRunCommand(CommandLine& cli) {
  cli.run.app = cli.app.add_subcommand("run", "Run local inference without a server");
  cli.run.app->fallthrough();
  AddModelOptions(*cli.run.app, cli.run.options.model, true);
  AddSamplingOptions(*cli.run.app, cli.run.options.sampling);
  AddResolverOptions(*cli.run.app, cli.run.options.resolver);
  cli.run.app->add_option("--prompt", cli.run.options.prompt, "Non-empty prompt text")
      ->check(NonEmptyValidator("prompt"))
      ->required();
  cli.run.app->add_option("--max-tokens", cli.run.options.max_tokens, "Maximum generated tokens")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();
}

void AddClientCommands(CommandLine& cli) {
  cli.client_chat.app = cli.app.add_subcommand("chat", "Chat with a running inferx server");
  cli.client_chat.app->fallthrough();
  cli.client_chat.app->add_option("--endpoint", cli.client_chat.options.endpoint, "Server base URL")
      ->check(NonEmptyValidator("endpoint"))
      ->capture_default_str();
  cli.client_chat.app->add_option("--model", cli.client_chat.options.model, "Served model name");
  CLI::Option* chat_prompt =
      cli.client_chat.app
          ->add_option("--prompt", cli.client_chat.options.prompt, "One-shot message")
          ->check(NonEmptyValidator("prompt"));
  CLI::Option* interactive = cli.client_chat.app->add_flag(
      "--interactive", cli.client_chat.options.interactive, "Start an interactive session");
  interactive->excludes(chat_prompt);

  cli.client_complete.app =
      cli.app.add_subcommand("complete", "Send a completion request to a running inferx server");
  cli.client_complete.app->fallthrough();
  cli.client_complete.app
      ->add_option("--endpoint", cli.client_complete.options.endpoint, "Server base URL")
      ->check(NonEmptyValidator("endpoint"))
      ->capture_default_str();
  cli.client_complete.app->add_option("--model", cli.client_complete.options.model,
                                      "Served model name");
  cli.client_complete.app
      ->add_option("--prompt", cli.client_complete.options.prompt, "Completion prompt")
      ->check(NonEmptyValidator("prompt"))
      ->required();
}

void AddInspectCommand(CommandLine& cli) {
  cli.inspect.app = cli.app.add_subcommand("inspect", "Inspect a model or compiled FSM metadata");
  cli.inspect.app->fallthrough();
  AddModelOptions(*cli.inspect.app, cli.inspect.options.model, false);
  AddResolverOptions(*cli.inspect.app, cli.inspect.options.resolver);
  CLI::Option* fsm_schema =
      cli.inspect.app->add_flag("--fsm-schema", cli.inspect.options.fsm_schema,
                                "Print the compiled request-state-machine schema as JSON");
  fsm_schema->excludes("--model");
}

void AddDownloadCommand(CommandLine& cli) {
  cli.download.app = cli.app.add_subcommand("download", "Prefetch a model into the local cache");
  cli.download.app->fallthrough();
  cli.download.app
      ->add_option("--model", cli.download.options.model, "Hugging Face model ID or local path")
      ->check(NonEmptyValidator("model"))
      ->required();
  AddResolverOptions(*cli.download.app, cli.download.options.resolver);
}

void AddInfoCommands(CommandLine& cli) {
  cli.version.app = cli.app.add_subcommand("version", "Print version and build information");
  cli.version.app->fallthrough();

  // Name matches vllm's `collect-env` subcommand, not a bare `env`.
  cli.collect_env.app = cli.app.add_subcommand(
      "collect-env", "Print PyTorch/CUDA/OS/driver and InferX build diagnostics");
  cli.collect_env.app->fallthrough();
}

void RegisterCommands(CommandLine& cli) {
  AddGlobalOptions(cli);
  AddServeCommand(cli);
  AddBenchCommands(cli);
  AddRunCommand(cli);
  AddClientCommands(cli);
  AddInspectCommand(cli);
  AddDownloadCommand(cli);
  AddInfoCommands(cli);
}

/// Parses argv into the registered options. A bare `inferx` prints top-level
/// help and is a usage exit; CLI11 help/version exits print their output and
/// report kTerminalOutputExit; every other parse failure reports kUsageExit.
/// Ok means exactly one subcommand was selected and SelectInvocation may run.
absl::Status ParseArguments(CommandLine& cli, int argc, const char* const* argv) {
  if (argc <= 1) {
    std::cout << cli.app.help();
    return absl::Status(kUsageExit, "inferx: a subcommand is required");
  }
  try {
    // The vector overloads expect arguments in reverse order (CLI11 pops
    // from the back); the const-argv overload handles that and the
    // program-name skip itself.
    cli.app.parse(argc, argv);
  } catch (const CLI::ParseError& parse_error) {
    const int exit_status = cli.app.exit(parse_error, std::cout, std::cerr);
    return exit_status == 0 ? absl::Status(kTerminalOutputExit, "terminal output printed")
                            : absl::Status(kUsageExit, "command line parse failed");
  }
  return absl::OkStatus();
}

/// Post-parse semantic validation: CLI11 checks syntax and ranges, this
/// covers cross-option rules whose messages name the command.
absl::Status ValidateSelection(std::string_view command, const command::SamplingOptions& sampling,
                               std::string_view message) {
  if (!ValidSampling(sampling)) return UsageError(command, message);
  return absl::OkStatus();
}

/// Selects the parsed subcommand's Invocation and applies the semantic
/// validation CLI11 cannot express.
StatusOr<command::Invocation> SelectInvocation(CommandLine& cli) {
  command::Invocation invocation;
  invocation.global = cli.global;

  if (cli.serve.app->parsed()) {
    if (!ValidSampling(cli.serve.options.sampling) ||
        !ValidProbability(cli.serve.options.gpu_memory_utilization, false)) {
      return UsageError("serve", "probabilities must be finite and in their documented ranges");
    }
    invocation.options = cli.serve.options;
  } else if (cli.bench_latency.app->parsed()) {
    if (absl::Status status = ValidateSelection("bench latency", cli.bench_latency.options.sampling,
                                                "invalid sampling values");
        !status.ok()) {
      return status;
    }
    invocation.options = cli.bench_latency.options;
  } else if (cli.bench_throughput.app->parsed()) {
    if (absl::Status status = ValidateSelection(
            "bench throughput", cli.bench_throughput.options.sampling, "invalid sampling values");
        !status.ok()) {
      return status;
    }
    invocation.options = cli.bench_throughput.options;
  } else if (cli.bench_serve.app->parsed()) {
    invocation.options = cli.bench_serve.options;
  } else if (cli.run.app->parsed()) {
    if (!ValidSampling(cli.run.options.sampling) || cli.run.options.prompt.empty()) {
      return UsageError("run", "prompt and sampling values are invalid");
    }
    invocation.options = cli.run.options;
  } else if (cli.client_chat.app->parsed()) {
    if (!cli.client_chat.options.interactive && cli.client_chat.options.prompt.empty()) {
      cli.client_chat.options.interactive = true;
    }
    invocation.options = cli.client_chat.options;
  } else if (cli.client_complete.app->parsed()) {
    invocation.options = cli.client_complete.options;
  } else if (cli.inspect.app->parsed()) {
    if (!cli.inspect.options.fsm_schema && cli.inspect.options.model.model.empty()) {
      return UsageError("inspect", "--model is required unless --fsm-schema is used");
    }
    invocation.options = cli.inspect.options;
  } else if (cli.download.app->parsed()) {
    invocation.options = cli.download.options;
  } else if (cli.version.app->parsed()) {
    invocation.options = command::VersionOptions{};
  } else if (cli.collect_env.app->parsed()) {
    invocation.options = command::CollectEnvOptions{};
  } else {
    return UsageError("inferx", "a subcommand is required");
  }

  return invocation;
}

}  // namespace

StatusOr<command::Invocation> ParseFromCommandLine(int argc, const char* const* argv) {
  CommandLine cli;
  RegisterCommands(cli);

  if (absl::Status parse = ParseArguments(cli, argc, argv); !parse.ok()) {
    return parse;
  }

  return SelectInvocation(cli);
}

void ConfigureLogging(const command::GlobalOptions& global) {
  absl::LogSeverity minimum = absl::LogSeverity::kInfo;
  int vlog_level = 0;

  switch (global.log_level) {
    case LogLevel::kTrace:
      vlog_level = 2;
      break;
    case LogLevel::kDebug:
      vlog_level = 1;
      break;
    case LogLevel::kInfo:
      break;
    case LogLevel::kWarning:
      minimum = absl::LogSeverity::kWarning;
      break;
    case LogLevel::kError:
      minimum = absl::LogSeverity::kError;
      break;
  }

  log::SetLevel(minimum, vlog_level);

  if (!global.log_file.empty() && !log::SetLogFile(global.log_file)) {
    LOG(ERROR) << "inferx: cannot open log file: " << global.log_file;
  }
}

}  // namespace inferx::cli