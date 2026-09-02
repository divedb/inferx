#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#include "inferx/artifacts/model_resolver.h"
#include "inferx/input/model_package.h"

namespace {

void PrintUsage(std::ostream& output, std::string_view program) {
  output << "usage: " << program
         << " MODEL [--revision REVISION] [--download-dir DIRECTORY] [--offline]\n"
         << "\n"
         << "MODEL is a local directory or a Hugging Face model ID such as "
            "Qwen/Qwen2.5-0.5B-Instruct. Local directories and the Hugging Face cache "
            "are checked before automatic download.\n";
}

struct Arguments {
  std::string model;
  inferx::artifacts::ModelResolverOptions resolver;
};

std::optional<Arguments> ParseArguments(int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--offline") {
      result.resolver.local_files_only = true;
    } else if (argument == "--revision" || argument == "--download-dir") {
      if (++index >= argc) {
        std::cerr << argument << " requires a value\n";
        return std::nullopt;
      }
      if (argument == "--revision") {
        result.resolver.revision = argv[index];
      } else {
        result.resolver.download_dir = std::filesystem::path(argv[index]);
      }
    } else if (argument.starts_with('-')) {
      std::cerr << "unknown option: " << argument << '\n';
      return std::nullopt;
    } else if (!result.model.empty()) {
      std::cerr << "only one MODEL may be specified\n";
      return std::nullopt;
    } else {
      result.model = argument;
    }
  }
  if (result.model.empty()) {
    std::cerr << "MODEL is required\n";
    return std::nullopt;
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    PrintUsage(std::cout, argv[0]);
    return 0;
  }
  auto arguments = ParseArguments(argc, argv);
  if (!arguments.has_value()) {
    PrintUsage(std::cerr, argv[0]);
    return 2;
  }
  inferx::artifacts::ModelResolver resolver;
  auto resolved = resolver.Resolve(arguments->model, arguments->resolver);
  if (!resolved.ok()) {
    std::cerr << "model resolution failed: " << resolved.status() << '\n';
    return 1;
  }
  inferx::input::PackageTokenizerPolicy policy;
  auto package = inferx::input::ModelPackageLoader::Load(resolved->path.string(), policy);
  if (!package.ok()) {
    std::cerr << "model inspection failed: " << package.status() << '\n';
    return 1;
  }
  const auto& llama = package->model_spec().llama();
  std::cout << "model=" << resolved->model << '\n'
            << "model_source=" << inferx::artifacts::ModelSourceName(resolved->source) << '\n'
            << "model_path=" << resolved->path.string() << '\n'
            << "model_revision=" << resolved->revision << '\n'
            << "architecture=llama\n"
            << "weights_entry=" << package->weights_entry().string() << '\n'
            << "vocab_size=" << llama.vocab_size << '\n'
            << "hidden_size=" << llama.hidden_size << '\n'
            << "layers=" << llama.num_hidden_layers << '\n'
            << "attention_heads=" << llama.num_attention_heads << '\n'
            << "key_value_heads=" << llama.num_key_value_heads << '\n'
            << "parameters_expected=" << package->weight_plan().coverage.expected << '\n'
            << "parameters_assigned=" << package->weight_plan().coverage.assigned << '\n'
            << "parameters_aliased=" << package->weight_plan().coverage.aliased << '\n'
            << "integrity_manifest=" << (package->integrity_manifest() ? "true" : "false") << '\n'
            << "tokenizer_qualified=true\n"
            << "model_fingerprint=" << package->fingerprint().digest().Hex() << '\n';
  return 0;
}
