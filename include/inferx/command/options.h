#ifndef INFERX_COMMAND_OPTIONS_H_
#define INFERX_COMMAND_OPTIONS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace inferx::command {

/// \brief Numeric precision used for model weights and activations.
enum class DType : uint8_t {
  kAuto,     ///< Infer dtype automatically from the model checkpoint.
  kFloat32,  ///< Full 32-bit floating point precision.
  kFloat16,  ///< IEEE 16-bit floating point (half precision).
  kBFloat16  ///< Brain floating point (16-bit, 8-bit exponent).
};

/// \brief Verbosity level for log output.
enum class LogLevel : uint8_t { kTrace, kDebug, kInfo, kWarning, kError };

/// \brief Output rendering format for command results.
enum class OutputFormat : uint8_t {
  kText,  ///< Human-readable plain text output.
  kJson   ///< Machine-readable JSON output.
};

/// \brief Weight/activation quantization scheme.
///
/// TODO(gc): Need to support quantization schemes.
/// EXAMPLE:
/// ```
/// ModelOptions model;
/// model.model = "TheBloke/Llama-2-7B-AWQ";
/// model.quantization = Quantization::kAwq;
/// model.dtype = DType::kFloat16;  // compute dtype for the dequantized activations
/// ```
enum class Quantization : uint8_t {
  kNone,          ///< No quantization; use `DType` as-is.
  kAuto,          ///< Infer from the checkpoint's quantization config.
  kAwq,           ///< Activation-aware Weight Quantization.
  kGptq,          ///< GPTQ post-training quantization.
  kFp8,           ///< FP8 weight/activation quantization.
  kBitsAndBytes,  ///< bitsandbytes int8/nf4 quantization.
  kGguf           ///< GGUF-format quantized weights.
};

/// \brief Data type used to store the KV cache.
///
/// TODO(gc): Need to support KV cache dtypes.
/// EXAMPLE:
/// ```
/// ModelOptions model;
/// model.model = "meta-llama/Llama-3.1-8B-Instruct";
/// model.dtype = DType::kBFloat16;             // compute/activation dtype
/// model.kv_cache_dtype = KVCacheDType::kFp8;  // shrink KV cache memory footprint
/// ```
enum class KVCacheDType : uint8_t {
  kAuto,  ///< Use the same dtype as the model's `DType`.
  kFp8    ///< Store the KV cache in FP8, trading accuracy for memory.
};

/// \brief On-disk format of the model checkpoint.
///
/// TODO(gc): Need to support checkpoint formats.
/// EXAMPLE:
/// ```
/// ModelOptions model;
/// model.model = "/local/models/llama-3-8b";
/// model.load_format = LoadFormat::kSafetensors;
/// ```
enum class LoadFormat : uint8_t {
  kAuto,         ///< Infer the format from the files present.
  kSafetensors,  ///< Load from `.safetensors` files.
  kPt,           ///< Load from PyTorch `.bin`/`.pt` checkpoint files.
  kNpcache,      ///< Load from a numpy cache of the checkpoint.
  kDummy,        ///< Skip loading weights; use random values (for testing).
  kTensorizer    ///< Load using the tensorizer serialization format.
};

/// \brief Options shared across all commands, controlling logging and
///        reproducibility.
struct GlobalOptions {
  /// Minimum severity of log messages that will be emitted.
  LogLevel log_level = LogLevel::kInfo;

  /// Path to a file to write logs to. If empty, logs go to stderr/stdout.
  std::string log_file;

  /// Seed for random number generation, used for reproducible runs.
  /// A value of 0 typically means "use a non-deterministic seed".
  uint64_t seed = 0;
};

/// \brief Options describing which model to load and how to run it.
struct ModelOptions {
  /// Path or identifier of the model to load (e.g. local path or hub repo).
  /// Corresponds to vllm's `--model`.
  std::string model;

  /// Path or identifier of the tokenizer to use. Defaults to the
  /// model's own tokenizer if left empty.
  /// Corresponds to vllm's `--tokenizer`.
  std::string tokenizer;

  /// Target device for inference (e.g. "auto", "cpu", "cuda:0").
  /// Corresponds to vllm's `--device`.
  std::string device = "auto";

  /// Numeric precision to load and run the model with.
  /// Corresponds to vllm's `--dtype`.
  DType dtype = DType::kAuto;

  /// Weight/activation quantization scheme.
  /// Corresponds to vllm's `--quantization`.
  Quantization quantization = Quantization::kNone;

  /// Data type used for the KV cache.
  /// Corresponds to vllm's `--kv-cache-dtype`.
  KVCacheDType kv_cache_dtype = KVCacheDType::kAuto;

  /// On-disk checkpoint format to load.
  /// Corresponds to vllm's `--load-format`.
  LoadFormat load_format = LoadFormat::kAuto;

  /// If true, allow executing custom modeling/tokenizer code shipped
  /// alongside the checkpoint (e.g. from the HF Hub).
  /// Corresponds to vllm's `--trust-remote-code`. Defaults to false for safety.
  bool trust_remote_code = false;

  /// Name the model is exposed as via the serving API, independent of
  /// the `model` path/identifier used to load it. If empty, `model` is
  /// used instead.
  /// Corresponds to vllm's `--served-model-name`.
  std::string served_model_name;

  /// Number of shards to split the model across for tensor parallelism.
  /// Corresponds to vllm's `--tensor-parallel-size`.
  uint32_t tensor_parallel_size = 1;

  /// Number of pipeline stages to split the model across for pipeline parallelism.
  /// Corresponds to vllm's `--pipeline-parallel-size`.
  uint32_t pipeline_parallel_size = 1;

  /// Maximum context length (in tokens) the model will be configured
  /// for. A value of 0 means "use the model's default maximum length".
  /// Corresponds to vllm's `--max-model-len`.
  uint64_t max_model_len = 0;

  /// Maximum sequence length for which CUDA graphs will be captured;
  /// longer sequences fall back to eager execution.
  /// Corresponds to vllm's `--max-seq-len-to-capture`.
  uint64_t max_seq_len_to_capture = 8192;

  /// If true, always run in eager (non-CUDA-graph) mode. Simpler and
  /// easier to debug, at some throughput cost.
  /// Corresponds to vllm's `--enforce-eager`.
  bool enforce_eager = false;

  /// Amount of CPU swap space, in GiB per GPU, used to offload KV
  /// cache blocks under memory pressure.
  /// Corresponds to vllm's `--swap-space`.
  double swap_space_gib = 4.0;

  /// Explicit override for the number of GPU KV-cache blocks to
  /// allocate, bypassing automatic profiling. Unset means "profile
  /// automatically".
  /// Corresponds to vllm's `--num-gpu-blocks-override`.
  std::optional<uint64_t> num_gpu_blocks_override;

  /// Maximum number of sequences that may be batched together at once.
  /// Corresponds to vllm's `--max-num-seqs`.
  uint32_t max_batch_size = 1;
};

/// \brief Options controlling how output tokens are sampled during
///        generation.
struct SamplingOptions {
  /// Sampling temperature. 0.0 selects greedy (deterministic) decoding.
  double temperature = 0.0;

  /// Nucleus (top-p) sampling threshold; only tokens within this
  /// cumulative probability mass are considered.
  double top_p = 1.0;

  /// Top-k sampling cutoff; only the k highest-probability tokens are
  /// considered. A value of 0 disables top-k filtering.
  uint32_t top_k = 0;

  /// Minimum token probability, relative to the most likely token, for
  /// a token to be considered.
  /// Corresponds to vllm's `min_p`.
  double min_p = 0.0;

  /// Penalizes tokens that have already appeared at all, regardless of
  /// how often. Positive values discourage repetition, negative values
  /// encourage it.
  /// Corresponds to vllm's `presence_penalty`.
  double presence_penalty = 0.0;

  /// Penalizes tokens proportionally to how often they've already
  /// appeared.
  /// Corresponds to vllm's `frequency_penalty`.
  double frequency_penalty = 0.0;

  /// Multiplicative penalty applied to logits of previously-seen
  /// tokens; 1.0 is a no-op.
  /// Corresponds to vllm's `repetition_penalty`.
  double repetition_penalty = 1.0;
};

/// \brief Options controlling how model/tokenizer artifacts are resolved and fetched from a remote
///        source.
struct ResolverOptions {
  /// Revision (branch, tag, or commit) of the model repository to use.
  /// Corresponds to vllm's `--revision`.
  std::string revision = "main";

  /// Revision of any custom modeling code shipped with the checkpoint,
  /// if different from `revision`.
  /// Corresponds to vllm's `--code-revision`.
  std::string code_revision;

  /// Local directory used to cache downloaded artifacts.
  /// Corresponds to vllm's `--download-dir`.
  std::string download_dir;

  /// If true, never attempt network access; only use locally cached files.
  /// Corresponds to running vllm with `HF_HUB_OFFLINE=1`.
  bool offline = false;
};

/// \brief Options for speculative decoding.
struct SpeculativeOptions {
  /// Draft model used to propose candidate tokens. Empty disables
  /// speculative decoding.
  std::string model;

  /// Number of tokens the draft model proposes per step.
  uint32_t num_speculative_tokens = 0;

  /// Maximum context length for the draft model. A value of 0 means
  /// "use the same value as the target model's `max_model_len`".
  uint64_t max_model_len = 0;
};

/// \brief Options for launching the inference server (`serve` command).
struct ServeOptions {
  /// Model to load and serve.
  ModelOptions model;

  /// Default sampling behavior for requests that don't override it.
  SamplingOptions sampling;

  /// Speculative decoding configuration. An empty `SpeculativeOptions::model` means speculative
  /// decoding is disabled.
  SpeculativeOptions speculative;

  /// Host/address the server binds to. Corresponds to vllm's `--host`.
  std::string host = "127.0.0.1";

  /// TCP port the server listens on. Corresponds to vllm's `--port`.
  uint16_t port = 8000;

  /// Unix domain socket path to serve over instead of TCP. If set,
  /// `host`/`port` are ignored. Corresponds to vllm's `--uds`.
  std::string uds;

  /// Number of API server processes to run.
  /// Corresponds to vllm's `--api-server-count`.
  uint32_t api_server_count = 1;

  /// If true, run without launching a local API server, for use in
  /// multi-node data-parallel deployments.
  /// Corresponds to vllm's `--headless`.
  bool headless = false;

  /// Path to a YAML file to load additional CLI options from.
  /// Corresponds to vllm's `--config`.
  std::string config_file;

  /// Maximum number of requests allowed to be running concurrently.
  /// Corresponds to vllm's `--max-num-seqs`.
  uint32_t max_running_requests = 256;

  /// Maximum number of tokens that may be batched together across all
  /// running requests in a single scheduler step.
  /// Corresponds to vllm's `--max-num-batched-tokens`.
  uint32_t max_num_batched_tokens = 2048;

  /// Fraction of total GPU memory the server is allowed to allocate.
  /// Corresponds to vllm's `--gpu-memory-utilization`.
  double gpu_memory_utilization = 0.9;

  /// If true, cache and reuse KV blocks for shared prompt prefixes
  /// across requests.
  /// Corresponds to vllm's `--enable-prefix-caching`.
  bool enable_prefix_caching = false;

  /// If true, split long prefills into chunks interleaved with decode
  /// steps rather than scheduling them as a single step.
  /// Corresponds to vllm's `--enable-chunked-prefill`.
  bool enable_chunked_prefill = false;

  /// Decoding-time backend used to enforce structured/guided output
  /// (e.g. "outlines", "xgrammar", "guidance"). Empty uses the
  /// server's default.
  /// Corresponds to vllm's `--guided-decoding-backend`.
  std::string guided_decoding_backend;

  /// If true, automatically select a tool-call parser based on the
  /// model and enable tool-call-triggered generation.
  /// Corresponds to vllm's `--enable-auto-tool-choice`.
  bool enable_auto_tool_choice = false;

  /// Name of the parser used to extract tool calls from model output
  /// (e.g. "hermes", "mistral", "llama3_json"). Only used when
  /// `enable_auto_tool_choice` is true.
  /// Corresponds to vllm's `--tool-call-parser`.
  std::string tool_call_parser;

  /// If true, enable LoRA adapter support.
  /// Corresponds to vllm's `--enable-lora`.
  bool enable_lora = false;

  /// Named LoRA adapters to make available at startup, given as
  /// "name=path" entries.
  /// Corresponds to vllm's `--lora-modules`.
  std::vector<std::string> lora_modules;

  /// Maximum number of LoRA adapters that may be loaded simultaneously.
  /// Corresponds to vllm's `--max-loras`.
  uint32_t max_loras = 1;

  /// Maximum rank supported for any loaded LoRA adapter.
  /// Corresponds to vllm's `--max-lora-rank`.
  uint32_t max_lora_rank = 16;

  /// Jinja2 chat template applied to chat-completion requests. May be
  /// a literal template string or a path to a template file. Empty
  /// uses the tokenizer's built-in template, if any.
  /// Corresponds to vllm's `--chat-template`.
  std::string chat_template;

  /// Bearer token required on incoming API requests. Empty disables
  /// authentication.
  /// Corresponds to vllm's `--api-key`.
  std::string api_key;

  /// Origins allowed by CORS.
  /// Corresponds to vllm's `--allowed-origins`.
  std::vector<std::string> allowed_origins = {"*"};

  /// HTTP methods allowed by CORS.
  /// Corresponds to vllm's `--allowed-methods`.
  std::vector<std::string> allowed_methods = {"*"};

  /// HTTP headers allowed by CORS.
  /// Corresponds to vllm's `--allowed-headers`.
  std::vector<std::string> allowed_headers = {"*"};

  /// If true, allow credentials (cookies, auth headers) on
  /// cross-origin requests.
  /// Corresponds to vllm's `--allow-credentials`.
  bool allow_credentials = false;

  /// Path to a PEM-encoded TLS private key. Enables HTTPS when set
  /// together with `ssl_certfile`.
  /// Corresponds to vllm's `--ssl-keyfile`.
  std::string ssl_keyfile;

  /// Path to a PEM-encoded TLS certificate.
  /// Corresponds to vllm's `--ssl-certfile`.
  std::string ssl_certfile;

  /// Path to a PEM-encoded CA certificate bundle used to verify client
  /// certificates.
  /// Corresponds to vllm's `--ssl-ca-certs`.
  std::string ssl_ca_certs;

  /// If true, suppress per-request logging; aggregate stats are still
  /// logged unless `disable_log_stats` is also set.
  /// Corresponds to vllm's `--disable-log-requests`.
  bool disable_log_requests = false;

  /// If true, suppress periodic aggregate statistics logging.
  /// Corresponds to vllm's `--disable-log-stats`.
  bool disable_log_stats = false;
};

/// \brief Which workload pattern a benchmark run should measure.
enum class BenchmarkMode : uint8_t {
  kLatency,     ///< Measure per-request latency for a single/small load.
  kThroughput,  ///< Measure maximum sustained throughput.
  kServe        ///< Benchmark against a running server endpoint.
};

/// \brief Options for running performance benchmarks (`benchmark`
///        command).
///
/// Field set is aligned with vllm's `vllm bench
/// latency|throughput|serve` subcommands.
struct BenchmarkOptions {
  /// Which benchmarking mode to run.
  BenchmarkMode mode = BenchmarkMode::kLatency;

  /// Model to benchmark (used for modes that load a model locally).
  ModelOptions model;

  /// Sampling parameters to use for generated benchmark requests.
  SamplingOptions sampling;

  /// Format used to print benchmark results.
  OutputFormat output_format = OutputFormat::kText;

  /// Server endpoint to benchmark against, used in kServe mode.
  /// Corresponds to vllm's `--host`/`--port` for `vllm bench serve`.
  std::string endpoint = "http://127.0.0.1:8000";

  /// Number of prompts/requests to issue during the benchmark.
  /// Corresponds to vllm's `--num-prompts`.
  uint32_t num_prompts = 1;

  /// Name of a built-in dataset to draw prompts from (e.g. "sharegpt",
  /// "random", "sonnet"). Corresponds to vllm's `--dataset-name`.
  std::string dataset_name = "random";

  /// Path to a dataset file, used when `dataset_name` requires one
  /// (e.g. "sharegpt"). Corresponds to vllm's `--dataset-path`.
  std::string dataset_path;

  /// Prompt length, in tokens, used when generating synthetic/random
  /// requests.
  /// Corresponds to vllm's `--random-input-len`.
  uint32_t random_input_len = 1024;

  /// Output length, in tokens, used when generating synthetic/random
  /// requests.
  /// Corresponds to vllm's `--random-output-len`.
  uint32_t random_output_len = 128;

  /// Average number of requests issued per second; requests are paced
  /// accordingly rather than fired all at once. A value of 0 means
  /// "send as fast as possible".
  /// Corresponds to vllm's `--request-rate`.
  double request_rate = 0.0;

  /// If true, capture and save a profiler trace during the benchmark.
  /// Corresponds to vllm's `--profile`.
  bool profile = false;

  /// If true, write detailed per-request results to disk in addition
  /// to the summary.
  /// Corresponds to vllm's `--save-result`.
  bool save_result = false;
};

/// \brief Options for a single local inference run (`run` command).
struct RunOptions {
  /// Model to load for the run.
  ModelOptions model;

  /// Sampling parameters to use when generating output.
  SamplingOptions sampling;

  /// Options for resolving/downloading the model artifacts.
  ResolverOptions resolver;

  /// Input prompt text to generate a completion for.
  std::string prompt;

  /// Maximum number of tokens to generate.
  uint32_t max_tokens = 8;
};

/// \brief Options for the interactive/one-shot client that talks to a
///        running server (`client` command).
struct ClientOptions {
  /// Server endpoint to connect to.
  std::string endpoint = "http://127.0.0.1:8000";

  /// Name/identifier of the model to request from the server.
  std::string model;

  /// Prompt to send in one-shot (non-interactive) mode.
  std::string prompt;

  /// If true, start an interactive REPL session instead of a single
  /// request.
  bool interactive = false;
};

/// \brief Options for inspecting a model without running inference
///        (`inspect` command).
struct InspectOptions {
  /// Model to inspect.
  ModelOptions model;

  /// Options for resolving/downloading the model artifacts.
  ResolverOptions resolver;

  /// If true, also print/derive the finite-state-machine (grammar)
  /// schema associated with the model, when applicable.
  bool fsm_schema = false;
};

/// \brief Options for downloading model artifacts ahead of time (`download` command).
struct DownloadOptions {
  /// Model to download.
  std::string model;

  /// Options controlling how the download is resolved and cached.
  ResolverOptions resolver;
};

}  // namespace inferx::command

#endif  // INFERX_COMMAND_OPTIONS_H_