#include <utility>

#include "tokenizer/adapters/adapter.h"
#include "tokenizer/internal/adapters/default_adapter.h"
#include "tokenizer/internal/adapters/llama_family_adapter.h"

namespace tokenizer {

AdapterRegistry::AdapterRegistry() : default_adapter_(std::make_shared<DefaultAdapter>()) {
  // Nothing is registered here.
  //
  // `LlamaFamilyAdapter` exists and is documented, but it is deliberately
  // *not* installed by default. It was written for the behaviour Transformers
  // 4.x had, where LlamaTokenizerFast rewrote the serialized post-processor
  // from `add_bos_token` at construction time. Transformers 5.x does not:
  // loading DeepSeek-V2-Lite there yields a TokenizersBackend whose
  // `add_bos_token` reads False even though its tokenizer_config.json says
  // true, and no BOS is prepended. Registering the adapter by default would
  // therefore make us disagree with the reference implementation on every
  // Llama-family checkpoint.
  //
  // A caller who needs the older behaviour opts in explicitly:
  //
  //   AdapterRegistry::Instance().Register(
  //       "LlamaTokenizerFast", std::make_shared<LlamaFamilyAdapter>());
  //
  // Keying is on `tokenizer_class` from tokenizer_config.json -- the same
  // signal Hugging Face dispatches on -- never on a model name, which a
  // directory rename would break.
}

AdapterRegistry& AdapterRegistry::Instance() {
  static AdapterRegistry* instance = new AdapterRegistry();
  return *instance;
}

void AdapterRegistry::Register(std::string tokenizer_class,
                               std::shared_ptr<const TokenizerAdapter> adapter) {
  adapters_[std::move(tokenizer_class)] = std::move(adapter);
}

std::shared_ptr<const TokenizerAdapter> AdapterRegistry::For(
    std::string_view tokenizer_class) const {
  auto it = adapters_.find(tokenizer_class);
  // A null entry means a caller unregistered the family again; fall back
  // rather than handing out a null adapter.
  if (it != adapters_.end() && it->second != nullptr) return it->second;
  return default_adapter_;
}

}  // namespace tokenizer
