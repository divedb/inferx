#include "inferx/cli/app.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/command/dispatcher.h"
#include "inferx/command/options.h"

namespace inferx::cli {
namespace {

class FakeDispatcher final : public command::Dispatcher {
 public:
  command::ExitCode Serve(const command::GlobalOptions& global,
                          const command::ServeOptions& options) override {
    called = "serve";
    global_options = global;
    serve_options = options;
    return result;
  }
  command::ExitCode Bench(const command::GlobalOptions& global,
                          const command::BenchmarkOptions& options) override {
    called = "bench";
    global_options = global;
    benchmark_options = options;
    return result;
  }
  command::ExitCode Run(const command::GlobalOptions&, const command::RunOptions&) override {
    called = "run";
    return result;
  }
  command::ExitCode Chat(const command::GlobalOptions&, const command::ClientOptions&) override {
    called = "chat";
    return result;
  }
  command::ExitCode Complete(const command::GlobalOptions&,
                             const command::ClientOptions&) override {
    called = "complete";
    return result;
  }
  command::ExitCode Inspect(const command::GlobalOptions&,
                            const command::InspectOptions&) override {
    called = "inspect";
    return result;
  }
  command::ExitCode Download(const command::GlobalOptions&,
                             const command::DownloadOptions&) override {
    called = "download";
    return result;
  }
  command::ExitCode Simulate(const command::GlobalOptions&,
                             const command::SimulateOptions&) override {
    called = "simulate";
    return result;
  }
  command::ExitCode Version(const command::GlobalOptions&) override {
    called = "version";
    return result;
  }
  command::ExitCode Environment(const command::GlobalOptions&) override {
    called = "env";
    return result;
  }

  std::string called;
  command::ExitCode result = command::ExitCode::kSuccess;
  command::GlobalOptions global_options;
  command::ServeOptions serve_options;
  command::BenchmarkOptions benchmark_options;
};

int RunArguments(std::vector<std::string> arguments, FakeDispatcher& dispatcher,
                 std::ostringstream& output, std::ostringstream& error) {
  std::vector<const char*> argv;
  argv.reserve(arguments.size());
  for (const std::string& argument : arguments) argv.push_back(argument.c_str());
  return Run(static_cast<int>(argv.size()), argv.data(), dispatcher, output, error);
}

TEST(CliAppTest, ServeMapsSharedAndGlobalConfigurationBeforeDispatch) {
  FakeDispatcher dispatcher;
  std::ostringstream output;
  std::ostringstream error;
  const int result = RunArguments({"inferx",
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
                                  dispatcher, output, error);

  EXPECT_EQ(result, 0);
  EXPECT_EQ(dispatcher.called, "serve");
  EXPECT_EQ(dispatcher.global_options.log_level, command::LogLevel::kDebug);
  EXPECT_EQ(dispatcher.global_options.log_file, "inferx.log");
  EXPECT_EQ(dispatcher.global_options.seed, 42U);
  EXPECT_EQ(dispatcher.serve_options.model.model, "model-id");
  EXPECT_EQ(dispatcher.serve_options.model.tokenizer, "tokenizer-id");
  EXPECT_EQ(dispatcher.serve_options.model.device, "cuda:1");
  EXPECT_EQ(dispatcher.serve_options.model.dtype, command::DType::kBFloat16);
  EXPECT_EQ(dispatcher.serve_options.model.tensor_parallel_size, 2U);
  EXPECT_EQ(dispatcher.serve_options.model.max_model_len, 8192U);
  EXPECT_EQ(dispatcher.serve_options.model.max_batch_size, 8U);
  EXPECT_DOUBLE_EQ(dispatcher.serve_options.sampling.temperature, 0.5);
  EXPECT_DOUBLE_EQ(dispatcher.serve_options.sampling.top_p, 0.9);
  EXPECT_EQ(dispatcher.serve_options.sampling.top_k, 20U);
  EXPECT_EQ(dispatcher.serve_options.host, "0.0.0.0");
  EXPECT_EQ(dispatcher.serve_options.port, 9000U);
  EXPECT_EQ(dispatcher.serve_options.max_running_requests, 64U);
  EXPECT_DOUBLE_EQ(dispatcher.serve_options.gpu_memory_utilization, 0.8);
}

TEST(CliAppTest, NestedBenchmarkDispatchesWithEnumOutputFormat) {
  FakeDispatcher dispatcher;
  std::ostringstream output;
  std::ostringstream error;
  const int result = RunArguments({"inferx", "bench", "throughput", "--model", "m",
                                   "--output-format", "JSON", "--requests", "12"},
                                  dispatcher, output, error);

  EXPECT_EQ(result, 0);
  EXPECT_EQ(dispatcher.called, "bench");
  EXPECT_EQ(dispatcher.benchmark_options.mode, command::BenchmarkMode::kThroughput);
  EXPECT_EQ(dispatcher.benchmark_options.output_format, command::OutputFormat::kJson);
  EXPECT_EQ(dispatcher.benchmark_options.requests, 12U);
}

TEST(CliAppTest, InvalidValuesNeverDispatch) {
  FakeDispatcher dispatcher;
  std::ostringstream output;
  std::ostringstream error;
  const int result =
      RunArguments({"inferx", "serve", "--model", "m", "--gpu-memory-utilization", "0"}, dispatcher,
                   output, error);

  EXPECT_EQ(result, static_cast<int>(command::ExitCode::kUsage));
  EXPECT_TRUE(dispatcher.called.empty());
  EXPECT_NE(error.str().find("error:"), std::string::npos);
}

TEST(CliAppTest, EveryCommandDispatchesThroughTheInjectedImplementation) {
  struct DispatchCase {
    std::vector<std::string> arguments;
    std::string expected;
  };
  const std::vector<DispatchCase> cases{
      {{"inferx", "serve", "--model", "m"}, "serve"},
      {{"inferx", "bench", "latency", "--model", "m"}, "bench"},
      {{"inferx", "run", "--model", "m", "--prompt", "hello"}, "run"},
      {{"inferx", "chat"}, "chat"},
      {{"inferx", "complete", "--prompt", "hello"}, "complete"},
      {{"inferx", "inspect", "--model", "m"}, "inspect"},
      {{"inferx", "download", "--model", "m"}, "download"},
      {{"inferx", "simulate", "validate-config", "--config", "config.json"}, "simulate"},
      {{"inferx", "version"}, "version"},
      {{"inferx", "env"}, "env"},
  };

  for (const DispatchCase& test_case : cases) {
    SCOPED_TRACE(test_case.expected);
    FakeDispatcher dispatcher;
    std::ostringstream output;
    std::ostringstream error;
    EXPECT_EQ(RunArguments(test_case.arguments, dispatcher, output, error), 0);
    EXPECT_EQ(dispatcher.called, test_case.expected);
  }
}

TEST(CliAppTest, DispatcherExitCodeIsPreserved) {
  FakeDispatcher dispatcher;
  dispatcher.result = command::ExitCode::kUnavailable;
  std::ostringstream output;
  std::ostringstream error;
  const int result = RunArguments({"inferx", "serve", "--model", "m"}, dispatcher, output, error);

  EXPECT_EQ(result, static_cast<int>(command::ExitCode::kUnavailable));
  EXPECT_EQ(dispatcher.called, "serve");
}

}  // namespace
}  // namespace inferx::cli
