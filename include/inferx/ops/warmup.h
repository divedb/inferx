#ifndef INFERX_OPS_WARMUP_H_
#define INFERX_OPS_WARMUP_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/ops/kernel_registry.h"

namespace inferx::ops {

struct LlamaOperatorSpec {
  uint64_t vocab_size = 0;
  uint64_t hidden_size = 0;
  uint64_t intermediate_size = 0;
  uint32_t num_attention_heads = 0;
  uint32_t num_key_value_heads = 0;
  uint32_t head_dim = 0;
};

struct OperatorEnvelope {
  DType storage_dtype = DType::kFloat32;
  uint32_t maximum_batch = 1;
  uint32_t maximum_prompt_tokens = 512;
  uint32_t maximum_context = 4096;
  uint64_t workspace_limit_bytes = 0;
  std::vector<uint32_t> token_buckets{1};
  uint16_t compute_capability = 0;
  DeviceKind device_kind = DeviceKind::kHost;
};

using PrepareKernel =
    std::function<absl::StatusOr<PreparedKernelPlan>(const KernelSelection&, const KernelKey&)>;

[[nodiscard]] absl::StatusOr<std::vector<KernelKey>> BuildRequiredKernelSet(
    const LlamaOperatorSpec& model, const OperatorEnvelope& envelope);
[[nodiscard]] absl::StatusOr<PreparedKernelCache> WarmupKernelSet(
    KernelRegistry& registry, std::span<const KernelKey> required_keys, size_t cache_capacity,
    const PrepareKernel& prepare, std::optional<BackendId> forced_backend = std::nullopt);

}  // namespace inferx::ops

#endif  // INFERX_OPS_WARMUP_H_
