#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
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
#include "inferx/base/log.h"
#include "inferx/base/token.h"
#include "common.h"
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

class DiscardingResponseSink final : public ResponseSink {
 public:
  absl::StatusOr<ResponseReservation> Prepare(ResponseEvent event) override {
    return ResponseReservation(std::move(event));
  }
  void Commit(ResponseReservation) noexcept override {}
};

}  // namespace

void Run(const GlobalOptions& global, const RunOptions& options) {
  if ((!options.model.tokenizer.empty()) ||
      (options.model.device != "auto" && options.model.device != "cpu") ||
      options.model.tensor_parallel_size != 1 ||
      (options.model.dtype != DType::kAuto && options.model.dtype != DType::kFloat32)) {
    LOG(ERROR) << "run: the current local runner supports the model tokenizer, one CPU, and float32";
    return;
  }
  if (options.sampling.temperature != 0.0 || options.sampling.top_p != 1.0 ||
      options.sampling.top_k != 0) {
    LOG(ERROR) << "run: the current backend supports deterministic greedy decoding only";
    return;
  }

  artifacts::ModelResolver resolver;
  auto resolved = resolver.Resolve(options.model.model, internal::ResolverOptions(options.resolver));
  if (!resolved.ok()) {
    internal::Fail("model resolution failed", resolved.status());
    return;
  }
  auto package = input::ModelPackageLoader::Load(resolved->path.string());
  if (!package.ok()) {
    internal::Fail("model loading failed", package.status());
    return;
  }
  const model::LlamaSpec& llama = package->model_spec().llama();
  if (llama.max_position_embeddings > (std::numeric_limits<uint32_t>::max)()) {
    internal::Fail("model loading failed",
                   absl::OutOfRangeError("model context exceeds the request token-count range"));
    return;
  }
  uint64_t context_capacity = llama.max_position_embeddings;
  if (options.model.max_model_len != 0) {
    context_capacity = (std::min)(context_capacity, options.model.max_model_len);
  }
  if (context_capacity > (std::numeric_limits<uint32_t>::max)()) {
    internal::Fail("configuration failed", absl::OutOfRangeError("max model length is too large"));
    return;
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
  if (!request.ok()) {
    internal::Fail("request validation failed", request.status());
    return;
  }
  const auto* prompt_tokens = std::get_if<std::vector<TokenId>>(&request->input);
  if (prompt_tokens == nullptr) {
    internal::Fail("inference failed",
                   absl::InternalError("prompt processor produced no token IDs"));
    return;
  }
  const std::vector<TokenId> prompt_token_ids = *prompt_tokens;

  runtime::CpuExecutionBackend backend;
  runtime::ModelLoadPlan load_plan;
  load_plan.package = &*package;
  load_plan.model_root = resolved->path.string();
  load_plan.max_prefill_tokens = prompt_token_ids.size();
  load_plan.context_capacity = context_capacity;
  auto handle = backend.Load(load_plan);
  if (!handle.ok()) {
    internal::Fail("backend model loading failed", handle.status());
    return;
  }

  DiscardingResponseSink sink;
  auto runner = runtime::SingleRequestRunner::Create(backend, *handle, load_plan.context_capacity,
                                                     load_plan.max_prefill_tokens, sink);
  if (!runner.ok()) {
    internal::Fail("inference setup failed", runner.status());
    return;
  }
  auto generated = (*runner)->Run(std::move(*request), MonotonicTime{});
  if (!generated.ok()) {
    internal::Fail("inference failed", generated.status());
    return;
  }
  if (!generated->status.ok()) {
    internal::Fail("inference failed", generated->status);
    return;
  }

  tokenization::DecodeOptions decode_options;
  decode_options.skip_special_tokens = true;
  auto decoded = package->tokenizer().Decode(generated->output_tokens, decode_options);
  if (!decoded.ok()) {
    internal::Fail("decode failed", decoded.status());
    return;
  }

  internal::kOut << "{\"model\":" << JsonString(resolved->model)
                 << ",\"model_path\":" << JsonString(resolved->path.string())
                 << ",\"model_revision\":" << JsonString(resolved->revision)
                 << ",\"seed\":" << global.seed
                 << ",\"temperature\":" << options.sampling.temperature
                 << ",\"prompt_token_ids\":";
  PrintTokenIds(internal::kOut, prompt_token_ids);
  internal::kOut << ",\"output_token_ids\":";
  PrintTokenIds(internal::kOut, generated->output_tokens);
  internal::kOut << ",\"output_text\":" << JsonString(*decoded)
                 << ",\"finish_reason\":" << JsonString(ToString(generated->finish)) << "}\n";

  runner->reset();
  const absl::Status unloaded = backend.Unload(*handle);
  if (!unloaded.ok()) internal::Fail("backend unload failed", unloaded);
}

}  // namespace inferx::command
