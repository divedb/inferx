#include "inferx/cli/app.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>

#include "CLI/CLI.hpp"
#include "inferx/base/version.h"
#include "inferx/command/options.h"

namespace inferx::cli {
namespace {

using command::BenchmarkMode;
using command::DType;
using command::ExitCode;
using command::LogLevel;
using command::OutputFormat;

constexpr int Code(ExitCode code) { return static_cast<int>(code); }

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

int UsageError(std::ostream& error, std::string_view command, std::string_view message) {
  error << "error: " << command << ": " << message << '\n';
  return Code(ExitCode::kUsage);
}

bool ValidProbability(double value, bool allow_zero) {
  return std::isfinite(value) && value <= 1.0 && (allow_zero ? value >= 0.0 : value > 0.0);
}

bool ValidSampling(const command::SamplingOptions& options) {
  return std::isfinite(options.temperature) && options.temperature >= 0.0 &&
         ValidProbability(options.top_p, false);
}

}  // namespace

int Run(int argc, const char* const* argv, command::Dispatcher& dispatcher, std::ostream& output,
        std::ostream& error) {
  command::GlobalOptions global;
  command::ServeOptions serve_options;
  command::BenchmarkOptions latency_options;
  command::BenchmarkOptions throughput_options;
  command::BenchmarkOptions serve_benchmark_options;
  command::RunOptions run_options;
  command::ClientOptions chat_options;
  command::ClientOptions complete_options;
  command::InspectOptions inspect_options;
  command::DownloadOptions download_options;

  CLI::App app{"InferX unified inference runtime", "inferx"};
  app.set_help_flag("-h,--help", "Show help and exit");
  app.set_version_flag("--version", std::string("inferx ") + std::string(GetVersionString()),
                       "Print the concise InferX version");
  app.failure_message([](const CLI::App* failed_app, const CLI::Error& failure) {
    return "error: " + failed_app->get_display_name() + ": " + failure.what() +
           "\nRun with --help for more information.\n";
  });
  app.positionals_at_end();
  app.require_subcommand(1);

  const std::map<std::string, LogLevel> log_levels{{"trace", LogLevel::kTrace},
                                                   {"debug", LogLevel::kDebug},
                                                   {"info", LogLevel::kInfo},
                                                   {"warning", LogLevel::kWarning},
                                                   {"error", LogLevel::kError}};
  app.add_option("--log-level", global.log_level, "Logging severity")
      ->transform(StrictEnumTransformer(log_levels))
      ->default_str("info");
  app.add_option("--log-file", global.log_file, "Write logs to this file");
  app.add_option("--seed", global.seed, "Deterministic unsigned 64-bit seed")
      ->capture_default_str();

  CLI::App* serve = app.add_subcommand("serve", "Start an inference server");
  EnableGlobalFallthrough(*serve);
  AddModelOptions(*serve, serve_options.model, true);
  AddSamplingOptions(*serve, serve_options.sampling);
  serve->add_option("--host", serve_options.host, "Listener host or address")
      ->check(NonEmptyValidator("host"))
      ->capture_default_str();
  serve->add_option("--port", serve_options.port, "Listener port")
      ->check(CLI::Range(uint16_t{1}, (std::numeric_limits<uint16_t>::max)()))
      ->capture_default_str();
  serve
      ->add_option("--max-running-requests", serve_options.max_running_requests,
                   "Maximum concurrently running requests")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();
  serve
      ->add_option("--gpu-memory-utilization", serve_options.gpu_memory_utilization,
                   "Fraction of GPU memory available to InferX")
      ->check(CLI::Range(0.0, 1.0))
      ->capture_default_str();

  CLI::App* bench = app.add_subcommand("bench", "Run inference performance benchmarks");
  bench->require_subcommand(1);
  EnableGlobalFallthrough(*bench);
  CLI::App* latency = bench->add_subcommand("latency", "Measure latency, TTFT, and TPOT");
  EnableGlobalFallthrough(*latency);
  latency_options.mode = BenchmarkMode::kLatency;
  AddModelOptions(*latency, latency_options.model, true);
  AddSamplingOptions(*latency, latency_options.sampling);
  AddOutputFormat(*latency, latency_options.output_format);
  latency->add_option("--requests", latency_options.num_prompts, "Measured request count")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();

  CLI::App* throughput =
      bench->add_subcommand("throughput", "Measure request and token throughput");
  EnableGlobalFallthrough(*throughput);
  throughput_options.mode = BenchmarkMode::kThroughput;
  AddModelOptions(*throughput, throughput_options.model, true);
  AddSamplingOptions(*throughput, throughput_options.sampling);
  AddOutputFormat(*throughput, throughput_options.output_format);
  throughput->add_option("--requests", throughput_options.num_prompts, "Measured request count")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();

  CLI::App* serve_benchmark =
      bench->add_subcommand("serve", "Benchmark a running inferx serve instance");
  EnableGlobalFallthrough(*serve_benchmark);
  serve_benchmark_options.mode = BenchmarkMode::kServe;
  serve_benchmark->add_option("--endpoint", serve_benchmark_options.endpoint, "Server base URL")
      ->check(NonEmptyValidator("endpoint"))
      ->capture_default_str();
  serve_benchmark->add_option("--model", serve_benchmark_options.model.model, "Served model name");
  AddOutputFormat(*serve_benchmark, serve_benchmark_options.output_format);
  serve_benchmark
      ->add_option("--requests", serve_benchmark_options.num_prompts, "Measured request count")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();

  CLI::App* run = app.add_subcommand("run", "Run local inference without a server");
  EnableGlobalFallthrough(*run);
  AddModelOptions(*run, run_options.model, true);
  AddSamplingOptions(*run, run_options.sampling);
  AddResolverOptions(*run, run_options.resolver);
  run->add_option("--prompt", run_options.prompt, "Non-empty prompt text")
      ->check(NonEmptyValidator("prompt"))
      ->required();
  run->add_option("--max-tokens", run_options.max_tokens, "Maximum generated tokens")
      ->check(CLI::Range(uint32_t{1}, (std::numeric_limits<uint32_t>::max)()))
      ->capture_default_str();

  CLI::App* chat = app.add_subcommand("chat", "Chat with a running inferx server");
  EnableGlobalFallthrough(*chat);
  chat->add_option("--endpoint", chat_options.endpoint, "Server base URL")
      ->check(NonEmptyValidator("endpoint"))
      ->capture_default_str();
  chat->add_option("--model", chat_options.model, "Served model name");
  CLI::Option* chat_prompt = chat->add_option("--prompt", chat_options.prompt, "One-shot message")
                                 ->check(NonEmptyValidator("prompt"));
  CLI::Option* interactive =
      chat->add_flag("--interactive", chat_options.interactive, "Start an interactive session");
  interactive->excludes(chat_prompt);

  CLI::App* complete =
      app.add_subcommand("complete", "Send a completion request to a running inferx server");
  EnableGlobalFallthrough(*complete);
  complete->add_option("--endpoint", complete_options.endpoint, "Server base URL")
      ->check(NonEmptyValidator("endpoint"))
      ->capture_default_str();
  complete->add_option("--model", complete_options.model, "Served model name");
  complete->add_option("--prompt", complete_options.prompt, "Completion prompt")
      ->check(NonEmptyValidator("prompt"))
      ->required();

  CLI::App* inspect = app.add_subcommand("inspect", "Inspect a model or compiled FSM metadata");
  EnableGlobalFallthrough(*inspect);
  AddModelOptions(*inspect, inspect_options.model, false);
  AddResolverOptions(*inspect, inspect_options.resolver);
  CLI::Option* fsm_schema =
      inspect->add_flag("--fsm-schema", inspect_options.fsm_schema,
                        "Print the compiled request-state-machine schema as JSON");
  fsm_schema->excludes("--model");

  CLI::App* download = app.add_subcommand("download", "Prefetch a model into the local cache");
  EnableGlobalFallthrough(*download);
  download->add_option("--model", download_options.model, "Hugging Face model ID or local path")
      ->check(NonEmptyValidator("model"))
      ->required();
  AddResolverOptions(*download, download_options.resolver);

  CLI::App* version = app.add_subcommand("version", "Print version and build information");
  EnableGlobalFallthrough(*version);
  CLI::App* environment = app.add_subcommand("env", "Print runtime environment diagnostics");
  EnableGlobalFallthrough(*environment);

  if (argc <= 1) {
    output << app.help();
    return Code(ExitCode::kUsage);
  }

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& parse_error) {
    const int result = app.exit(parse_error, output, error);
    return result == 0 ? Code(ExitCode::kSuccess) : Code(ExitCode::kUsage);
  }

  if (serve->parsed()) {
    if (!ValidSampling(serve_options.sampling) ||
        !ValidProbability(serve_options.gpu_memory_utilization, false)) {
      return UsageError(error, "serve",
                        "probabilities must be finite and in their documented ranges");
    }
    return Code(dispatcher.Serve(global, serve_options));
  }
  if (latency->parsed()) {
    if (!ValidSampling(latency_options.sampling)) {
      return UsageError(error, "bench latency", "invalid sampling values");
    }
    return Code(dispatcher.Bench(global, latency_options));
  }
  if (throughput->parsed()) {
    if (!ValidSampling(throughput_options.sampling)) {
      return UsageError(error, "bench throughput", "invalid sampling values");
    }
    return Code(dispatcher.Bench(global, throughput_options));
  }
  if (serve_benchmark->parsed()) {
    return Code(dispatcher.Bench(global, serve_benchmark_options));
  }
  if (run->parsed()) {
    if (!ValidSampling(run_options.sampling) || run_options.prompt.empty()) {
      return UsageError(error, "run", "prompt and sampling values are invalid");
    }
    return Code(dispatcher.Run(global, run_options));
  }
  if (chat->parsed()) {
    if (!chat_options.interactive && chat_options.prompt.empty()) chat_options.interactive = true;
    return Code(dispatcher.Chat(global, chat_options));
  }
  if (complete->parsed()) return Code(dispatcher.Complete(global, complete_options));
  if (inspect->parsed()) {
    if (!inspect_options.fsm_schema && inspect_options.model.model.empty()) {
      return UsageError(error, "inspect", "--model is required unless --fsm-schema is used");
    }
    return Code(dispatcher.Inspect(global, inspect_options));
  }
  if (download->parsed()) return Code(dispatcher.Download(global, download_options));
  if (version->parsed()) return Code(dispatcher.Version(global));
  if (environment->parsed()) return Code(dispatcher.Environment(global));
  return UsageError(error, "inferx", "a subcommand is required");
}

}  // namespace inferx::cli
