#include "inferx/cli/app.h"

#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/command/options.h"

namespace inferx::cli {
namespace {

using command::BenchmarkMode;
using command::LogLevel;
using command::OutputFormat;

StatusOr<command::Invocation> Parse(const std::vector<std::string>& arguments) {
  std::vector<const char*> argv;
  argv.reserve(arguments.size());
  for (const std::string& argument : arguments) argv.push_back(argument.c_str());
  return ParseFromCommandLine(static_cast<int>(argv.size()), argv.data());
}

TEST(CliParseTest, ServeMapsGlobalAndCommandOptions) {
  auto result = Parse({"inferx",
                       "--log-level",
                       "debug",
                       "--log-file",
                       "inferx.log",
                       "--seed",
                       "42",
                       "serve",
                       "--model",
                       "model-id",
                       "--tokenizer",
                       "tokenizer-id",
                       "--device",
                       "cuda:1",
                       "--dtype",
                       "bfloat16",
                       "--tensor-parallel-size",
                       "2",
                       "--max-model-len",
                       "8192",
                       "--max-batch-size",
                       "8",
                       "--temperature",
                       "0.5",
                       "--top-p",
                       "0.9",
                       "--top-k",
                       "20",
                       "--host",
                       "0.0.0.0",
                       "--port",
                       "9000",
                       "--max-running-requests",
                       "64",
                       "--gpu-memory-utilization",
                       "0.8"});

  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->global.log_level, LogLevel::kDebug);
  EXPECT_EQ(result->global.log_file, "inferx.log");
  EXPECT_EQ(result->global.seed, 42U);
  ASSERT_TRUE(std::holds_alternative<command::ServeOptions>(result->options));
  const auto& serve = std::get<command::ServeOptions>(result->options);
  EXPECT_EQ(serve.model.model, "model-id");
  EXPECT_EQ(serve.model.tokenizer, "tokenizer-id");
  EXPECT_EQ(serve.model.device, "cuda:1");
  EXPECT_EQ(serve.model.dtype, command::DType::kBFloat16);
  EXPECT_EQ(serve.model.tensor_parallel_size, 2U);
  EXPECT_EQ(serve.model.max_model_len, 8192U);
  EXPECT_EQ(serve.model.max_batch_size, 8U);
  EXPECT_DOUBLE_EQ(serve.sampling.temperature, 0.5);
  EXPECT_DOUBLE_EQ(serve.sampling.top_p, 0.9);
  EXPECT_EQ(serve.sampling.top_k, 20U);
  EXPECT_EQ(serve.host, "0.0.0.0");
  EXPECT_EQ(serve.port, 9000U);
  EXPECT_EQ(serve.max_running_requests, 64U);
  EXPECT_DOUBLE_EQ(serve.gpu_memory_utilization, 0.8);
}

TEST(CliParseTest, BenchThroughputKeepsModeFormatAndRequestCount) {
  auto result = Parse(
      {"inferx", "benchmark", "throughput", "--model", "m", "--output-format", "JSON", "--num-prompts", "12"});

  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(std::holds_alternative<command::BenchmarkOptions>(result->options));
  const auto& bench = std::get<command::BenchmarkOptions>(result->options);
  EXPECT_EQ(bench.mode, BenchmarkMode::kThroughput);
  EXPECT_EQ(bench.output_format, OutputFormat::kJson);
  EXPECT_EQ(bench.num_prompts, 12U);
}

TEST(CliParseTest, BenchNestedSubcommandsSelectTheirMode) {
  auto latency = Parse({"inferx", "benchmark", "latency", "--model", "m"});
  ASSERT_TRUE(latency.ok()) << latency.status();
  EXPECT_EQ(std::get<command::BenchmarkOptions>(latency->options).mode, BenchmarkMode::kLatency);

  auto serve = Parse({"inferx", "benchmark", "serve", "--endpoint", "http://127.0.0.1:9"});
  ASSERT_TRUE(serve.ok()) << serve.status();
  EXPECT_EQ(std::get<command::BenchmarkOptions>(serve->options).mode, BenchmarkMode::kServe);
}

TEST(CliParseTest, EveryCommandParsesToItsOptionsAndName) {
  struct ParseCase {
    std::vector<std::string> arguments;
    size_t option_index;
    std::string_view command_name;
  };
  const std::vector<ParseCase> cases{
      {{"inferx", "serve", "--model", "m"}, 0, "serve"},
      {{"inferx", "benchmark", "latency", "--model", "m"}, 1, "benchmark"},
      {{"inferx", "run", "--model", "m", "--prompt", "hello"}, 2, "run"},
      {{"inferx", "chat"}, 3, "chat"},
      {{"inferx", "version"}, 4, "version"},
      {{"inferx", "collect-env"}, 5, "collect-env"},
  };

  for (const ParseCase& test_case : cases) {
    SCOPED_TRACE(test_case.command_name);
    auto result = Parse(test_case.arguments);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result->options.index(), test_case.option_index);
    EXPECT_EQ(CommandName(*result), test_case.command_name);
  }
}

TEST(CliParseTest, InvalidSamplingIsAUsageError) {
  auto result = Parse({"inferx", "serve", "--model", "m", "--gpu-memory-utilization", "0"});
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnknown);
}

TEST(CliParseTest, MissingPromptIsAUsageError) {
  auto result = Parse({"inferx", "run", "--model", "m"});
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnknown);
}

TEST(CliParseTest, ChatDefaultsToInteractiveWithoutPrompt) {
  auto plain = Parse({"inferx", "chat"});
  ASSERT_TRUE(plain.ok()) << plain.status();
  EXPECT_TRUE(std::get<command::ChatOptions>(plain->options).interactive);

  auto prompted = Parse({"inferx", "chat", "--prompt", "hi"});
  ASSERT_TRUE(prompted.ok()) << prompted.status();
  const auto& chat = std::get<command::ChatOptions>(prompted->options);
  EXPECT_FALSE(chat.interactive);
  EXPECT_EQ(chat.prompt, "hi");
}

TEST(CliParseTest, TerminalExitsDoNotProduceAnInvocation) {
  // Help and --version print their output and report kCancelled (exit 0);
  // a bare invocation and parse failures report kUnknown (exit 2, kUsage).
  auto help = Parse({"inferx", "--help"});
  EXPECT_FALSE(help.ok());
  EXPECT_EQ(help.status().code(), absl::StatusCode::kCancelled);

  auto version = Parse({"inferx", "--version"});
  EXPECT_FALSE(version.ok());
  EXPECT_EQ(version.status().code(), absl::StatusCode::kCancelled);

  auto no_arguments = Parse({"inferx"});
  EXPECT_FALSE(no_arguments.ok());
  EXPECT_EQ(no_arguments.status().code(), absl::StatusCode::kUnknown);

  auto bad_flag = Parse({"inferx", "--no-such-flag", "version"});
  EXPECT_FALSE(bad_flag.ok());
  EXPECT_EQ(bad_flag.status().code(), absl::StatusCode::kUnknown);
}

}  // namespace
}  // namespace inferx::cli
