#ifndef INFERX_COMMAND_OPTIONS_H_
#define INFERX_COMMAND_OPTIONS_H_

#include <cstdint>
#include <optional>
#include <string>

namespace inferx::command {

enum class DType : uint8_t { kAuto, kFloat32, kFloat16, kBFloat16 };
enum class LogLevel : uint8_t { kTrace, kDebug, kInfo, kWarning, kError };
enum class OutputFormat : uint8_t { kText, kJson };

struct GlobalOptions {
  LogLevel log_level = LogLevel::kInfo;
  std::string log_file;
  uint64_t seed = 0;
};

struct ModelOptions {
  std::string model;
  std::string tokenizer;
  std::string device = "auto";
  DType dtype = DType::kAuto;
  uint32_t tensor_parallel_size = 1;
  uint64_t max_model_len = 0;
  uint32_t max_batch_size = 1;
};

struct SamplingOptions {
  double temperature = 0.0;
  double top_p = 1.0;
  uint32_t top_k = 0;
};

struct ResolverOptions {
  std::string revision = "main";
  std::string download_dir;
  bool offline = false;
};

struct ServeOptions {
  ModelOptions model;
  SamplingOptions sampling;
  std::string host = "127.0.0.1";
  uint16_t port = 8000;
  uint32_t max_running_requests = 256;
  double gpu_memory_utilization = 0.9;
};

enum class BenchmarkMode : uint8_t { kLatency, kThroughput, kServe };

struct BenchmarkOptions {
  BenchmarkMode mode = BenchmarkMode::kLatency;
  ModelOptions model;
  SamplingOptions sampling;
  OutputFormat output_format = OutputFormat::kText;
  std::string endpoint = "http://127.0.0.1:8000";
  uint32_t requests = 1;
};

struct RunOptions {
  ModelOptions model;
  SamplingOptions sampling;
  ResolverOptions resolver;
  std::string prompt;
  uint32_t max_tokens = 8;
};

struct ClientOptions {
  std::string endpoint = "http://127.0.0.1:8000";
  std::string model;
  std::string prompt;
  bool interactive = false;
};

struct InspectOptions {
  ModelOptions model;
  ResolverOptions resolver;
  bool fsm_schema = false;
};

struct DownloadOptions {
  std::string model;
  ResolverOptions resolver;
};

enum class SimulateOperation : uint8_t {
  kValidateConfig,
  kExplainConfig,
  kRun,
  kReplay,
  kCheckTrace,
};

struct SimulateOptions {
  SimulateOperation operation = SimulateOperation::kValidateConfig;
  std::string config;
  std::string workload;
  std::string trace;
  std::string output;
  bool overwrite = false;
  bool verbose = false;

#define INFERX_COMMAND_OVERRIDE_FIELD(name) std::optional<uint64_t> name;
  INFERX_COMMAND_OVERRIDE_FIELD(max_active_sequences)
  INFERX_COMMAND_OVERRIDE_FIELD(max_model_tokens)
  INFERX_COMMAND_OVERRIDE_FIELD(max_output_tokens)
  INFERX_COMMAND_OVERRIDE_FIELD(max_prompt_tokens)
  INFERX_COMMAND_OVERRIDE_FIELD(max_queued_requests)
  INFERX_COMMAND_OVERRIDE_FIELD(max_scheduled_tokens_per_step)
  INFERX_COMMAND_OVERRIDE_FIELD(max_sequences_per_step)
  INFERX_COMMAND_OVERRIDE_FIELD(max_simulation_events)
  INFERX_COMMAND_OVERRIDE_FIELD(plan_buffer_slots)
  INFERX_COMMAND_OVERRIDE_FIELD(response_channel_capacity)
  INFERX_COMMAND_OVERRIDE_FIELD(simulated_kv_token_capacity)
  INFERX_COMMAND_OVERRIDE_FIELD(submission_channel_capacity)
  INFERX_COMMAND_OVERRIDE_FIELD(fake_base_latency_ns)
  INFERX_COMMAND_OVERRIDE_FIELD(fake_prefill_latency_per_token_ns)
  INFERX_COMMAND_OVERRIDE_FIELD(fake_decode_latency_per_sequence_ns)
#undef INFERX_COMMAND_OVERRIDE_FIELD
};

}  // namespace inferx::command

#endif  // INFERX_COMMAND_OPTIONS_H_
