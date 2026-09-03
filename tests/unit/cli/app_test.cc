#include "inferx/cli/app.h"

#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/command/options.h"

namespace inferx::cli {
namespace {

using command::BenchmarkMode;
using command::Command;
using command::LogLevel;
using command::OutputFormat;

ParseResult Parse(const std::vector<std::string>& arguments, std::ostringstream& output,
                  std::ostringstream& error) {
  std::vector<const char*> argv;
  argv.reserve(arguments.size());
  for (const std::string& argument : arguments) argv.push_back(argument.c_str());
  return ParseFromCommandLine(static_cast<int>(argv.size()), argv.data(), output, error);
}

TEST(CliParseTest, ServeMapsGlobalAndCommandOptions) {
  std::ostringstream output;
  std::ostringstream error;
  const ParseResult result = Parse({"inferx",
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
                                    "0.8"},
                                   output, error);

  EXPECT_TRUE(result.should_run);
  EXPECT_EQ(result.exit_code, command::ExitCode::kSuccess);
  ASSERT_EQ(result.invocation.command, Command::kServe);
  EXPECT_EQ(result.invocation.global.log_level, LogLevel::kDebug);
  EXPECT_EQ(result.invocation.global.log_file, "inferx.log");
  EXPECT_EQ(result.invocation.global.seed, 42U);
  const auto& serve = std::get<command::ServeOptions>(result.invocation.options);
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
  std::ostringstream output;
  std::ostringstream error;
  const ParseResult result = Parse(
      {"inferx", "bench", "throughput", "--model", "m", "--output-format", "JSON", "--requests", "12"},
      output, error);

  EXPECT_TRUE(result.should_run);
  ASSERT_EQ(result.invocation.command, Command::kBench);
  const auto& bench = std::get<command::BenchmarkOptions>(result.invocation.options);
  EXPECT_EQ(bench.mode, BenchmarkMode::kThroughput);
  EXPECT_EQ(bench.output_format, OutputFormat::kJson);
  EXPECT_EQ(bench.num_prompts, 12U);
}

TEST(CliParseTest, BenchNestedSubcommandsSelectTheirMode) {
  std::ostringstream output;
  std::ostringstream error;
  const ParseResult latency =
      Parse({"inferx", "bench", "latency", "--model", "m"}, output, error);
  ASSERT_EQ(latency.invocation.command, Command::kBench);
  EXPECT_EQ(std::get<command::BenchmarkOptions>(latency.invocation.options).mode,
            BenchmarkMode::kLatency);

  const ParseResult serve =
      Parse({"inferx", "bench", "serve", "--endpoint", "http://127.0.0.1:9"}, output, error);
  ASSERT_EQ(serve.invocation.command, Command::kBench);
  EXPECT_EQ(std::get<command::BenchmarkOptions>(serve.invocation.options).mode,
            BenchmarkMode::kServe);
}

TEST(CliParseTest, EveryCommandParsesToItsInvocation) {
  struct DispatchCase {
    std::vector<std::string> arguments;
    Command command;
  };
  const std::vector<DispatchCase> cases{
      {{"inferx", "serve", "--model", "m"}, Command::kServe},
      {{"inferx", "bench", "latency", "--model", "m"}, Command::kBench},
      {{"inferx", "run", "--model", "m", "--prompt", "hello"}, Command::kRun},
      {{"inferx", "chat"}, Command::kChat},
      {{"inferx", "complete", "--prompt", "hello"}, Command::kComplete},
      {{"inferx", "inspect", "--model", "m"}, Command::kInspect},
      {{"inferx", "download", "--model", "m"}, Command::kDownload},
      {{"inferx", "version"}, Command::kVersion},
      {{"inferx", "env"}, Command::kEnvironment},
  };

  for (const DispatchCase& test_case : cases) {
    SCOPED_TRACE(test_case.arguments.back());
    std::ostringstream output;
    std::ostringstream error;
    const ParseResult result = Parse(test_case.arguments, output, error);
    EXPECT_EQ(result.exit_code, command::ExitCode::kSuccess);
    EXPECT_TRUE(result.should_run);
    EXPECT_EQ(result.invocation.command, test_case.command);
    if (test_case.command == Command::kVersion || test_case.command == Command::kEnvironment) {
      EXPECT_TRUE(std::holds_alternative<std::monostate>(result.invocation.options));
    }
  }
}

TEST(CliParseTest, InvalidValuesNeverRun) {
  std::ostringstream output;
  std::ostringstream error;
  const ParseResult result =
      Parse({"inferx", "serve", "--model", "m", "--gpu-memory-utilization", "0"}, output, error);

  EXPECT_FALSE(result.should_run);
  EXPECT_EQ(result.exit_code, command::ExitCode::kUsage);
  EXPECT_NE(error.str().find("error:"), std::string::npos);
}

TEST(CliParseTest, MissingPromptIsAUsageError) {
  std::ostringstream output;
  std::ostringstream error;
  const ParseResult result = Parse({"inferx", "run", "--model", "m"}, output, error);

  EXPECT_FALSE(result.should_run);
  EXPECT_EQ(result.exit_code, command::ExitCode::kUsage);
  EXPECT_NE(error.str().find("error:"), std::string::npos);
}

TEST(CliParseTest, InspectRequiresModelOrFsmSchema) {
  std::ostringstream output;
  std::ostringstream error;
  const ParseResult result = Parse({"inferx", "inspect"}, output, error);

  EXPECT_FALSE(result.should_run);
  EXPECT_EQ(result.exit_code, command::ExitCode::kUsage);
  EXPECT_NE(error.str().find("--model is required unless --fsm-schema is used"), std::string::npos);
}

TEST(CliParseTest, ChatDefaultsToInteractiveWithoutPrompt) {
  std::ostringstream output;
  std::ostringstream error;
  const ParseResult plain = Parse({"inferx", "chat"}, output, error);
  ASSERT_EQ(plain.invocation.command, Command::kChat);
  EXPECT_TRUE(std::get<command::ClientOptions>(plain.invocation.options).interactive);

  const ParseResult prompted = Parse({"inferx", "chat", "--prompt", "hi"}, output, error);
  const auto& chat = std::get<command::ClientOptions>(prompted.invocation.options);
  EXPECT_FALSE(chat.interactive);
  EXPECT_EQ(chat.prompt, "hi");
}

TEST(CliParseTest, HelpAndVersionExitsDoNotRun) {
  std::ostringstream output;
  std::ostringstream error;
  const ParseResult help = Parse({"inferx", "--help"}, output, error);
  EXPECT_FALSE(help.should_run);
  EXPECT_EQ(help.exit_code, command::ExitCode::kSuccess);
  EXPECT_NE(output.str().find("InferX unified inference runtime"), std::string::npos);

  output.str("");
  const ParseResult version = Parse({"inferx", "--version"}, output, error);
  EXPECT_FALSE(version.should_run);
  EXPECT_EQ(version.exit_code, command::ExitCode::kSuccess);
  EXPECT_NE(output.str().find("inferx"), std::string::npos);

  output.str("");
  const ParseResult no_arguments = Parse({"inferx"}, output, error);
  EXPECT_FALSE(no_arguments.should_run);
  EXPECT_EQ(no_arguments.exit_code, command::ExitCode::kUsage);
  EXPECT_NE(output.str().find("InferX unified inference runtime"), std::string::npos);
}

}  // namespace
}  // namespace inferx::cli
