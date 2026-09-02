#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/api/response_event.h"
#include "inferx/artifacts/model_resolver.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/engine/event_sink.h"
#include "inferx/input/model_package.h"
#include "inferx/input/prompt_input.h"
#include "inferx/input/prompt_processor.h"
#include "inferx/runtime/cpu_execution_backend.h"
#include "inferx/runtime/execution_backend.h"
#include "inferx/runtime/single_request_runner.h"
#include "inferx/tokenization/tokenizer_options.h"

namespace {

constexpr double kSupportedTemperature = 0.0;

struct Arguments {
  std::string model;
  std::string prompt;
  uint32_t max_tokens = 8;
  double temperature = kSupportedTemperature;
  inferx::artifacts::ModelResolverOptions resolver;
};

class DiscardingResponseSink final : public inferx::ResponseSink {
 public:
  absl::StatusOr<inferx::ResponseReservation> Prepare(inferx::ResponseEvent event) override {
    return inferx::ResponseReservation(std::move(event));
  }

  void Commit(inferx::ResponseReservation) noexcept override {}
};

void PrintUsage(std::ostream& output, std::string_view program) {
  output << "usage: " << program
         << " MODEL --prompt TEXT [--max-tokens N] [--temperature 0.0]"
            " [--revision REVISION] [--download-dir DIRECTORY] [--offline]\n"
         << "\n"
         << "Runs greedy Llama inference with the CPU reference backend and emits JSON. "
            "MODEL may be a local directory or a Hugging Face repository ID.\n";
}

template <typename Number>
bool ParseNumber(std::string_view text, Number& result) {
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, result);
  return parsed.ec == std::errc() && parsed.ptr == end;
}

std::optional<Arguments> ParseArguments(int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--offline") {
      result.resolver.local_files_only = true;
      continue;
    }
    if (argument == "--prompt" || argument == "--max-tokens" || argument == "--temperature" ||
        argument == "--revision" || argument == "--download-dir") {
      if (++index >= argc) {
        std::cerr << argument << " requires a value\n";
        return std::nullopt;
      }
      const std::string_view value(argv[index]);
      if (argument == "--prompt") {
        result.prompt = value;
      } else if (argument == "--max-tokens") {
        if (!ParseNumber(value, result.max_tokens) || result.max_tokens == 0) {
          std::cerr << "--max-tokens must be a positive 32-bit integer\n";
          return std::nullopt;
        }
      } else if (argument == "--temperature") {
        if (!ParseNumber(value, result.temperature) ||
            result.temperature != kSupportedTemperature) {
          std::cerr << "--temperature must be 0.0; the current backend uses greedy decoding\n";
          return std::nullopt;
        }
      } else if (argument == "--revision") {
        result.resolver.revision = value;
      } else {
        result.resolver.download_dir = std::filesystem::path(value);
      }
      continue;
    }
    if (argument.starts_with('-')) {
      std::cerr << "unknown option: " << argument << '\n';
      return std::nullopt;
    }
    if (!result.model.empty()) {
      std::cerr << "only one MODEL may be specified\n";
      return std::nullopt;
    }
    result.model = argument;
  }
  if (result.model.empty()) {
    std::cerr << "MODEL is required\n";
    return std::nullopt;
  }
  if (result.prompt.empty()) {
    std::cerr << "--prompt must be nonempty\n";
    return std::nullopt;
  }
  return result;
}

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

void PrintTokenIds(std::span<const inferx::TokenId> tokens) {
  std::cout << '[';
  for (size_t index = 0; index < tokens.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << tokens[index].value();
  }
  std::cout << ']';
}

absl::Status Run(const Arguments& arguments) {
  inferx::artifacts::ModelResolver resolver;
  auto resolved = resolver.Resolve(arguments.model, arguments.resolver);
  if (!resolved.ok()) return resolved.status();

  auto package = inferx::input::ModelPackageLoader::Load(resolved->path.string());
  if (!package.ok()) return package.status();
  const inferx::model::LlamaSpec& llama = package->model_spec().llama();
  if (llama.max_position_embeddings > std::numeric_limits<uint32_t>::max()) {
    return absl::OutOfRangeError("model context exceeds the request token-count range");
  }

  inferx::input::InputProcessingRequest input{inferx::RequestId(1),
                                              inferx::RequestEpoch(1),
                                              inferx::ModelId(0),
                                              inferx::input::RawTextPrompt{arguments.prompt, true},
                                              inferx::TokenCount(arguments.max_tokens),
                                              std::nullopt};
  const inferx::input::PromptModelFacts facts{
      &package->tokenizer(), llama.vocab_size,
      inferx::TokenCount(static_cast<uint32_t>(llama.max_position_embeddings))};
  auto request = inferx::input::PromptProcessor::Process(
      input, facts, inferx::input::PromptLimits{}, inferx::MonotonicTime{});
  if (!request.ok()) return request.status();
  const auto* prompt_tokens = std::get_if<std::vector<inferx::TokenId>>(&request->input);
  if (prompt_tokens == nullptr) {
    return absl::InternalError("prompt processor did not produce token IDs");
  }
  const std::vector<inferx::TokenId> prompt_token_ids = *prompt_tokens;

  inferx::runtime::CpuExecutionBackend backend;
  inferx::runtime::ModelLoadPlan load_plan;
  load_plan.package = &*package;
  load_plan.model_root = resolved->path.string();
  load_plan.max_prefill_tokens = prompt_token_ids.size();
  load_plan.context_capacity = llama.max_position_embeddings;
  auto handle = backend.Load(load_plan);
  if (!handle.ok()) return handle.status();

  DiscardingResponseSink sink;
  auto runner = inferx::runtime::SingleRequestRunner::Create(
      backend, *handle, load_plan.context_capacity, load_plan.max_prefill_tokens, sink);
  if (!runner.ok()) return runner.status();
  auto generated = (*runner)->Run(std::move(*request), inferx::MonotonicTime{});
  if (!generated.ok()) return generated.status();
  if (!generated->status.ok()) return generated->status;

  inferx::tokenization::DecodeOptions decode_options;
  decode_options.skip_special_tokens = true;
  auto decoded = package->tokenizer().Decode(generated->output_tokens, decode_options);
  if (!decoded.ok()) return decoded.status();

  std::cout << "{\"model\":" << JsonString(resolved->model)
            << ",\"model_path\":" << JsonString(resolved->path.string())
            << ",\"model_revision\":" << JsonString(resolved->revision)
            << ",\"temperature\":" << arguments.temperature << ",\"prompt_token_ids\":";
  PrintTokenIds(prompt_token_ids);
  std::cout << ",\"output_token_ids\":";
  PrintTokenIds(generated->output_tokens);
  std::cout << ",\"output_text\":" << JsonString(*decoded)
            << ",\"finish_reason\":" << JsonString(inferx::ToString(generated->finish)) << "}\n";

  runner->reset();
  return backend.Unload(*handle);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    PrintUsage(std::cout, argv[0]);
    return 0;
  }
  const std::optional<Arguments> arguments = ParseArguments(argc, argv);
  if (!arguments.has_value()) {
    PrintUsage(std::cerr, argv[0]);
    return 2;
  }
  const absl::Status status = Run(*arguments);
  if (!status.ok()) {
    std::cerr << "generation failed: " << status << '\n';
    return 1;
  }
  return 0;
}
