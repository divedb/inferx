#include <algorithm>
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
#include "absl/strings/str_cat.h"
#include "common.h"
#include "inferx/api/generate_request.h"
#include "inferx/api/response_event.h"
#include "inferx/artifacts/artifact_tensor.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/log.h"
#include "inferx/base/status_macros.h"
#include "inferx/base/token.h"
#include "inferx/engine/event_sink.h"
#include "inferx/input/model_package.h"
#include "inferx/input/prompt_input.h"
#include "inferx/input/prompt_processor.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/runtime/cpu_execution_backend.h"
#include "inferx/runtime/execution_backend.h"
#include "inferx/runtime/single_request_runner.h"
#include "inferx/tokenization/tokenizer_options.h"

namespace inferx::command {
namespace {

// ---------------------------------------------------------------------------
// Output formatting.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Pipeline stages. Each returns a status whose message carries the failing
// stage's scope; Run logs the composed status once.
// ---------------------------------------------------------------------------

/// The local runner supports exactly: the model's own tokenizer, one CPU,
/// and float32 weights.
absl::Status ValidateSupportedConfiguration(const RunOptions& options) {
  const ModelOptions& model = options.model;
  if (!model.tokenizer.empty() || (model.device != "auto" && model.device != "cpu") ||
      model.tensor_parallel_size != 1 ||
      (model.dtype != DType::kAuto && model.dtype != DType::kFloat32)) {
    return absl::InvalidArgumentError(
        "the current local runner supports the model tokenizer, one CPU, and float32");
  }
  if (options.sampling.temperature != 0.0 || options.sampling.top_p != 1.0 ||
      options.sampling.top_k != 0) {
    return absl::InvalidArgumentError(
        "the current backend supports deterministic greedy decoding only");
  }
  return absl::OkStatus();
}

/// A resolved artifact package plus the effective context capacity.
struct LoadedModel {
  std::string model;
  std::string path;
  std::string revision;
  input::ValidatedModelPackage package;
  uint32_t context_capacity = 0;
};

absl::StatusOr<LoadedModel> ResolveModel(const RunOptions& options) {
  artifacts::ModelResolver resolver;
  auto resolved =
      resolver.Resolve(options.model.model, internal::ResolverOptions(options.resolver));
  if (!resolved.ok()) {
    return absl::Status(resolved.status().code(),
                        absl::StrCat("model resolution failed: ", resolved.status().message()));
  }
  auto package = input::ModelPackageLoader::Load(resolved->path.string());
  if (!package.ok()) {
    return absl::Status(package.status().code(),
                        absl::StrCat("model loading failed: ", package.status().message()));
  }
  const model::LlamaSpec& llama = package->model_spec().llama();
  if (llama.max_position_embeddings > (std::numeric_limits<uint32_t>::max)()) {
    return absl::OutOfRangeError("model context exceeds the request token-count range");
  }
  uint64_t capacity = llama.max_position_embeddings;
  if (options.model.max_model_len != 0)
    capacity = (std::min)(capacity, options.model.max_model_len);
  if (capacity > (std::numeric_limits<uint32_t>::max)()) {
    return absl::OutOfRangeError("max model length is too large");
  }
  return LoadedModel{resolved->model, resolved->path.string(), resolved->revision,
                     std::move(*package), static_cast<uint32_t>(capacity)};
}

/// A validated generation request plus a copy of its prompt token IDs
/// (the request itself is moved into the runner later).
struct PreparedRequest {
  GenerateRequest request;
  std::vector<TokenId> prompt_ids;
};

absl::StatusOr<PreparedRequest> PrepareRequest(const RunOptions& options,
                                               const LoadedModel& loaded) {
  const model::LlamaSpec& llama = loaded.package.model_spec().llama();
  input::InputProcessingRequest input_request{RequestId(1),
                                              RequestEpoch(1),
                                              ModelId(0),
                                              input::RawTextPrompt{options.prompt, true},
                                              TokenCount(options.max_tokens),
                                              std::nullopt};
  const input::PromptModelFacts facts{&loaded.package.tokenizer(), llama.vocab_size,
                                      TokenCount(loaded.context_capacity)};
  auto request =
      input::PromptProcessor::Process(input_request, facts, input::PromptLimits{}, MonotonicTime{});
  if (!request.ok()) {
    return absl::Status(request.status().code(),
                        absl::StrCat("request validation failed: ", request.status().message()));
  }
  const auto* prompt_tokens = std::get_if<std::vector<TokenId>>(&request->input);
  if (prompt_tokens == nullptr) {
    return absl::InternalError("prompt processor produced no token IDs");
  }
  return PreparedRequest{std::move(*request), *prompt_tokens};
}

/// Generated output tokens plus their decoded text.
struct GenerationResult {
  std::vector<TokenId> output_tokens;
  std::string output_text;
  FinishReason finish = FinishReason::kLength;
};

/// Result of one completed request; discards streamed deltas.
class DiscardingResponseSink final : public ResponseSink {
 public:
  absl::StatusOr<ResponseReservation> Prepare(ResponseEvent event) override {
    return ResponseReservation(std::move(event));
  }
  void Commit(ResponseReservation) noexcept override {}
};

absl::StatusOr<GenerationResult> Generate(const LoadedModel& loaded, PreparedRequest& prepared) {
  runtime::CpuExecutionBackend backend;
  runtime::ModelLoadPlan load_plan;
  load_plan.package = &loaded.package;
  load_plan.model_root = loaded.path;
  load_plan.max_prefill_tokens = prepared.prompt_ids.size();
  load_plan.context_capacity = loaded.context_capacity;
  auto handle = backend.Load(load_plan);
  if (!handle.ok()) {
    return absl::Status(handle.status().code(),
                        absl::StrCat("backend model loading failed: ", handle.status().message()));
  }

  DiscardingResponseSink sink;
  std::unique_ptr<runtime::SingleRequestRunner> runner;
  INFERX_ASSIGN_OR_RETURN(
      runner, runtime::SingleRequestRunner::Create(backend, *handle, load_plan.context_capacity,
                                                   load_plan.max_prefill_tokens, sink));
  auto generated = runner->Run(std::move(prepared.request), MonotonicTime{});
  if (!generated.ok()) {
    return absl::Status(generated.status().code(),
                        absl::StrCat("inference failed: ", generated.status().message()));
  }
  if (!generated->status.ok()) {
    return absl::Status(generated->status.code(),
                        absl::StrCat("inference failed: ", generated->status.message()));
  }

  GenerationResult result;
  result.output_tokens = generated->output_tokens;
  result.finish = generated->finish;

  tokenization::DecodeOptions decode_options;
  decode_options.skip_special_tokens = true;
  auto decoded = loaded.package.tokenizer().Decode(result.output_tokens, decode_options);
  if (!decoded.ok()) {
    return absl::Status(decoded.status().code(),
                        absl::StrCat("decode failed: ", decoded.status().message()));
  }
  result.output_text = std::move(*decoded);

  runner = nullptr;  // the runner borrows the backend; destroy it before Unload
  const absl::Status unloaded = backend.Unload(*handle);
  if (!unloaded.ok()) internal::Fail("backend unload failed", unloaded);
  return result;
}

void PrintResult(const GlobalOptions& global, const RunOptions& options, const LoadedModel& loaded,
                 const std::vector<TokenId>& prompt_token_ids, const GenerationResult& result) {
  internal::kOut << "{\"model\":" << JsonString(loaded.model)
                 << ",\"model_path\":" << JsonString(loaded.path)
                 << ",\"model_revision\":" << JsonString(loaded.revision)
                 << ",\"seed\":" << global.seed
                 << ",\"temperature\":" << options.sampling.temperature << ",\"prompt_token_ids\":";
  PrintTokenIds(internal::kOut, prompt_token_ids);
  internal::kOut << ",\"output_token_ids\":";
  PrintTokenIds(internal::kOut, result.output_tokens);
  internal::kOut << ",\"output_text\":" << JsonString(result.output_text)
                 << ",\"finish_reason\":" << JsonString(ToString(result.finish)) << "}\n";
}

absl::Status RunPipeline(const GlobalOptions& global, const RunOptions& options) {
  INFERX_RETURN_IF_ERROR(ValidateSupportedConfiguration(options));

  auto loaded = ResolveModel(options);
  if (!loaded.ok()) return loaded.status();

  auto prepared = PrepareRequest(options, *loaded);
  if (!prepared.ok()) return prepared.status();

  auto result = Generate(*loaded, *prepared);
  if (!result.ok()) return result.status();

  PrintResult(global, options, *loaded, prepared->prompt_ids, *result);
  return absl::OkStatus();
}

}  // namespace

void Run(const GlobalOptions& global, const RunOptions& options) {
  if (absl::Status status = RunPipeline(global, options); !status.ok()) {
    internal::Fail("run", status);
  }
}

}  // namespace inferx::command
