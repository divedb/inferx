#include <filesystem>
#include <iostream>
#include <string_view>

#include "inferx/model/model_package.h"

namespace {

void PrintUsage(std::string_view program) {
  std::cerr << "usage: " << program << " MODEL_DIRECTORY\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    PrintUsage(argv[0]);
    return 2;
  }
  inferx::model::ModelArtifactLoader loader;
  auto package = loader.Inspect(std::filesystem::path(argv[1]));
  if (!package.ok()) {
    std::cerr << "model inspection failed: " << package.status() << '\n';
    return 1;
  }
  const auto& llama = package->model_spec.llama();
  std::cout << "architecture=llama\n"
            << "weights_entry=" << package->weights_entry.string() << '\n'
            << "vocab_size=" << llama.vocab_size << '\n'
            << "hidden_size=" << llama.hidden_size << '\n'
            << "layers=" << llama.num_hidden_layers << '\n'
            << "attention_heads=" << llama.num_attention_heads << '\n'
            << "key_value_heads=" << llama.num_key_value_heads << '\n'
            << "parameters_expected=" << package->weight_plan.coverage.expected << '\n'
            << "parameters_assigned=" << package->weight_plan.coverage.assigned << '\n'
            << "parameters_aliased=" << package->weight_plan.coverage.aliased << '\n'
            << "integrity_manifest=" << (package->integrity_manifest ? "true" : "false") << '\n'
            << "tokenizer_qualified=false\n"
            << "model_fingerprint=unavailable\n";
  std::cerr << "note: tokenizer/fingerprint publication is blocked by the "
               "ADR 0025 dependency qualification gate\n";
  return 0;
}
