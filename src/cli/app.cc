#include "inferx/cli/app.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>

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
}

void AddSamplingOptions(CLI::App& app, command::SamplingOptions& options) {
  app.add_option("--temperature", options.temperature,
                 "Sampling temperature; 0 selects deterministic greedy decoding")
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();
  app.add_option("--top-p", options.top_p, "Nucleus sampling probability in (0, 1]")
      ->check(CLI::Range(0.0, 1.0))
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

void EnableGlobalFallthrough(CLI::App& app) { app.fallthrough(); }

bool ValidProbability(double value, bool allow_zero) {
  return std::isfinite(value) && value <= 1.0 && (allow_zero ? value >= 0.0 : value > 0.0);
}

bool ValidSampling(const command::SamplingOptions& options) {
  return std::isfinite(options.temperature) && options.temperature >= 0.0 &&
         ValidProbability(options.top_p, false);
}

absl::Status UsageError(std::string_view command, std::string_view message) {
  std::cerr << "error: " << command << ": " << message << '\n';
  return absl::Status(kUsageExit, std::string(command) + ": " + std::string(message));
}

/// The CLI11 application plus the option storage it binds. One instance
/// per ParseFromCommandLine call; the subcommand pointers record which
/// nested commands were registered so SelectInvocation can ask CLI11
/// which one the user actually selected.
struct CommandLine {
  command::GlobalOptions global;
  command::ServeOptions serve;
  command::BenchmarkOptions latency;
  command::BenchmarkOptions throughput;
  command::BenchmarkOptions serve_benchmark;
  command::RunOptions run;
  command::ClientOptions chat;
  command::ClientOptions complete;
  command::InspectOptions inspect;
  command::DownloadOptions download;

  CLI::App app{"InferX unified inference runtime", "inferx"};
  CLI::App* serve_command = nullptr;
  CLI::App* latency_command = nullptr;
  CLI::App* throughput_command = nullptr;
  CLI::App* serve_benchmark_command = nullptr;
  CLI::App* run_command = nullptr;
  CLI::App* chat_command = nullptr;
  CLI::App* complete_command = nullptr;
  CLI::App* inspect_command = nullptr;
  CLI::App* download_command = nullptr;
  CLI::App* version_command = nullptr;
  CLI::App* environment_command = nullptr;
};

void AddGlobalOptions(CommandLine& cli) {
  cli.app.set_help_flag("-h,--help", "Show help and exit");
  cli.app.set_version_flag(
      "--version", std::string("inferx ") + std::string(GetVersionString()),
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
  cli.serve_command = cli.app.add_subcommand("serve", "Start an inference server");
  EnableGlobalFallthrough(*cli.serve_command);
  AddModelOptions(*cli.serve_command, cli.serve.model, true);
  AddSamplingOptions(*cli.serve_command, cli.serve.sampling);
  cli.serve_command
      ->add_option("--host", cli.serve.host, "Listener host or address")
      ->check(NonEmptyValidator("host"))
      ->capture_default_str();
  cli.serve_command
      ->add_option("--port", cli.serve.port, "Listener port")
      ->check(CLI::Range(uint16_t{1}, (std::numeric_limits<uint16_t>::max)()))
      ->capture_default_str();
  cli.serve_command
      ->add_option("--max-running-requests", cli.serve.max_running_requests,
                   "Maximum concurrently running requests")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();
  cli.serve_command
      ->add_option("--gpu-memory-utilization", cli.serve.gpu_memory_utilization,
                   "Fraction of GPU memory available to InferX")
      ->check(CLI::Range(0.0, 1.0))
      ->capture_default_str();
}

void AddBenchCommands(CommandLine& cli) {
  CLI::App* bench = cli.app.add_subcommand("bench", "Run inference performance benchmarks");
  bench->require_subcommand(1);
  EnableGlobalFallthrough(*bench);

  cli.latency_command = bench->add_subcommand("latency", "Measure latency, TTFT, and TPOT");
  EnableGlobalFallthrough(*cli.latency_command);
  cli.latency.mode = BenchmarkMode::kLatency;
  AddModelOptions(*cli.latency_command, cli.latency.model, true);
  AddSamplingOptions(*cli.latency_command, cli.latency.sampling);
  AddOutputFormat(*cli.latency_command, cli.latency.output_format);
  cli.latency_command
      ->add_option("--requests", cli.latency.num_prompts, "Measured request count")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();

  cli.throughput_command =
      bench->add_subcommand("throughput", "Measure request and token throughput");
  EnableGlobalFallthrough(*cli.throughput_command);
  cli.throughput.mode = BenchmarkMode::kThroughput;
  AddModelOptions(*cli.throughput_command, cli.throughput.model, true);
  AddSamplingOptions(*cli.throughput_command, cli.throughput.sampling);
  AddOutputFormat(*cli.throughput_command, cli.throughput.output_format);
  cli.throughput_command
      ->add_option("--requests", cli.throughput.num_prompts, "Measured request count")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();

  cli.serve_benchmark_command =
      bench->add_subcommand("serve", "Benchmark a running inferx serve instance");
  EnableGlobalFallthrough(*cli.serve_benchmark_command);
  cli.serve_benchmark.mode = BenchmarkMode::kServe;
  cli.serve_benchmark_command
      ->add_option("--endpoint", cli.serve_benchmark.endpoint, "Server base URL")
      ->check(NonEmptyValidator("endpoint"))
      ->capture_default_str();
  cli.serve_benchmark_command->add_option("--model", cli.serve_benchmark.model.model,
                                          "Served model name");
  AddOutputFormat(*cli.serve_benchmark_command, cli.serve_benchmark.output_format);
  cli.serve_benchmark_command
      ->add_option("--requests", cli.serve_benchmark.num_prompts, "Measured request count")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();
}

void AddRunCommand(CommandLine& cli) {
  cli.run_command = cli.app.add_subcommand("run", "Run local inference without a server");
  EnableGlobalFallthrough(*cli.run_command);
  AddModelOptions(*cli.run_command, cli.run.model, true);
  AddSamplingOptions(*cli.run_command, cli.run.sampling);
  AddResolverOptions(*cli.run_command, cli.run.resolver);
  cli.run_command
      ->add_option("--prompt", cli.run.prompt, "Non-empty prompt text")
      ->check(NonEmptyValidator("prompt"))
      ->required();
  cli.run_command
      ->add_option("--max-tokens", cli.run.max_tokens, "Maximum generated tokens")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();
}

void AddClientCommands(CommandLine& cli) {
  cli.chat_command = cli.app.add_subcommand("chat", "Chat with a running inferx server");
  EnableGlobalFallthrough(*cli.chat_command);
  cli.chat_command
      ->add_option("--endpoint", cli.chat.endpoint, "Server base URL")
      ->check(NonEmptyValidator("endpoint"))
      ->capture_default_str();
  cli.chat_command->add_option("--model", cli.chat.model, "Served model name");
  CLI::Option* chat_prompt =
      cli.chat_command->add_option("--prompt", cli.chat.prompt, "One-shot message")
          ->check(NonEmptyValidator("prompt"));
  CLI::Option* interactive = cli.chat_command->add_flag(
      "--interactive", cli.chat.interactive, "Start an interactive session");
  interactive->excludes(chat_prompt);

  cli.complete_command =
      cli.app.add_subcommand("complete", "Send a completion request to a running inferx server");
  EnableGlobalFallthrough(*cli.complete_command);
  cli.complete_command
      ->add_option("--endpoint", cli.complete.endpoint, "Server base URL")
      ->check(NonEmptyValidator("endpoint"))
      ->capture_default_str();
  cli.complete_command->add_option("--model", cli.complete.model, "Served model name");
  cli.complete_command
      ->add_option("--prompt", cli.complete.prompt, "Completion prompt")
      ->check(NonEmptyValidator("prompt"))
      ->required();
}

void AddInspectCommand(CommandLine& cli) {
  cli.inspect_command =
      cli.app.add_subcommand("inspect", "Inspect a model or compiled FSM metadata");
  EnableGlobalFallthrough(*cli.inspect_command);
  AddModelOptions(*cli.inspect_command, cli.inspect.model, false);
  AddResolverOptions(*cli.inspect_command, cli.inspect.resolver);
  CLI::Option* fsm_schema = cli.inspect_command->add_flag(
      "--fsm-schema", cli.inspect.fsm_schema,
      "Print the compiled request-state-machine schema as JSON");
  fsm_schema->excludes("--model");
}

void AddDownloadCommand(CommandLine& cli) {
  cli.download_command =
      cli.app.add_subcommand("download", "Prefetch a model into the local cache");
  EnableGlobalFallthrough(*cli.download_command);
  cli.download_command
      ->add_option("--model", cli.download.model, "Hugging Face model ID or local path")
      ->check(NonEmptyValidator("model"))
      ->required();
  AddResolverOptions(*cli.download_command, cli.download.resolver);
}

void AddInfoCommands(CommandLine& cli) {
  cli.version_command =
      cli.app.add_subcommand("version", "Print version and build information");
  EnableGlobalFallthrough(*cli.version_command);
  cli.environment_command =
      cli.app.add_subcommand("env", "Print runtime environment diagnostics");
  EnableGlobalFallthrough(*cli.environment_command);
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

  if (cli.serve_command->parsed()) {
    if (!ValidSampling(cli.serve.sampling) ||
        !ValidProbability(cli.serve.gpu_memory_utilization, false)) {
      return UsageError("serve", "probabilities must be finite and in their documented ranges");
    }
    invocation.options = cli.serve;
  } else if (cli.latency_command->parsed()) {
    if (absl::Status status = ValidateSelection("bench latency", cli.latency.sampling,
                                                "invalid sampling values");
        !status.ok()) {
      return status;
    }
    invocation.options = cli.latency;
  } else if (cli.throughput_command->parsed()) {
    if (absl::Status status = ValidateSelection("bench throughput", cli.throughput.sampling,
                                                "invalid sampling values");
        !status.ok()) {
      return status;
    }
    invocation.options = cli.throughput;
  } else if (cli.serve_benchmark_command->parsed()) {
    invocation.options = cli.serve_benchmark;
  } else if (cli.run_command->parsed()) {
    if (!ValidSampling(cli.run.sampling) || cli.run.prompt.empty()) {
      return UsageError("run", "prompt and sampling values are invalid");
    }
    invocation.options = cli.run;
  } else if (cli.chat_command->parsed()) {
    if (!cli.chat.interactive && cli.chat.prompt.empty()) cli.chat.interactive = true;
    invocation.options = cli.chat;
  } else if (cli.complete_command->parsed()) {
    invocation.options = cli.complete;
  } else if (cli.inspect_command->parsed()) {
    if (!cli.inspect.fsm_schema && cli.inspect.model.model.empty()) {
      return UsageError("inspect", "--model is required unless --fsm-schema is used");
    }
    invocation.options = cli.inspect;
  } else if (cli.download_command->parsed()) {
    invocation.options = cli.download;
  } else if (cli.version_command->parsed()) {
    invocation.options = command::VersionOptions{};
  } else if (cli.environment_command->parsed()) {
    invocation.options = command::EnvironmentOptions{};
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
  log::Severity severity = log::Severity::kInfo;
  switch (global.log_level) {
    case LogLevel::kTrace:
      severity = log::Severity::kTrace;
      break;
    case LogLevel::kDebug:
      severity = log::Severity::kDebug;
      break;
    case LogLevel::kInfo:
      severity = log::Severity::kInfo;
      break;
    case LogLevel::kWarning:
      severity = log::Severity::kWarning;
      break;
    case LogLevel::kError:
      severity = log::Severity::kError;
      break;
  }
  log::SetMinSeverity(severity);
  if (!global.log_file.empty() && !log::SetFileSink(global.log_file)) {
    LOG(ERROR) << "inferx: cannot open log file: " << global.log_file;
  }
}

}  // namespace inferx::cli
