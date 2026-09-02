// Immutable effective configuration. Constructed
// only through ValidateConfig; no mutators, no environment reads after
// creation. CanonicalJson() emits the canonical UTF-8 form (schema version
// first, lexicographic field order, decimal integers, no whitespace) that is
// embedded byte-for-byte in replay headers.

#ifndef INFERX_CONFIG_ENGINE_CONFIG_H_
#define INFERX_CONFIG_ENGINE_CONFIG_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/base/token.h"
#include "inferx/config/parsed_config.h"

namespace inferx::config {

struct BuildCapabilities {
  bool simulator = true;
  bool cuda = false;
};

struct ModelCapabilities {
  inferx::TokenCount max_context_tokens;
  bool accepts_text = false;
};

class EngineConfig {
 public:
  // Validates ranges, cross-field rules, capabilities, and latency overflow;
  // reports independent field errors in stable (lexicographic) order.
  static absl::StatusOr<EngineConfig> Validate(const ParsedConfig& parsed,
                                               const BuildCapabilities& build,
                                               const ModelCapabilities& model);

  EngineConfig(const EngineConfig&) = default;
  EngineConfig& operator=(const EngineConfig&) = default;

#define INFERX_CONFIG_ACCESSOR(camel, json_name, default_value) \
  [[nodiscard]] uint64_t camel() const noexcept { return values_.camel.value; }

  INFERX_CONFIG_FIELDS(INFERX_CONFIG_ACCESSOR)
  INFERX_CUDA_CONFIG_FIELDS(INFERX_CONFIG_ACCESSOR)
#undef INFERX_CONFIG_ACCESSOR

  [[nodiscard]] bool has_cuda_section() const noexcept { return has_cuda_section_; }

  // Byte-stable canonical serialization (see file comment).
  [[nodiscard]] std::string CanonicalJson() const;

 private:
  EngineConfig() = default;
  ParsedConfig values_;
  bool has_cuda_section_ = false;
};

// Exposed for tests and tools: the field list with json spellings, sorted
// lexicographically — canonical writers iterate this order.
std::vector<std::string> CanonicalFieldOrder();

}  // namespace inferx::config

#endif  // INFERX_CONFIG_ENGINE_CONFIG_H_
