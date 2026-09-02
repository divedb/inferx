#ifndef INFERX_OPS_KERNEL_KEY_H_
#define INFERX_OPS_KERNEL_KEY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/ops/op_context.h"
#include "inferx/tensor/dtype.h"

namespace inferx::ops {

struct KernelKey {
  static constexpr uint16_t kSchemaVersion = 1;
  static constexpr size_t kMaxDimensions = 8;

  uint16_t schema_version = kSchemaVersion;
  OpKind op = OpKind::kEmbedding;
  ExecutionPhase phase = ExecutionPhase::kGeneric;
  DeviceKind device_kind = DeviceKind::kHost;
  uint16_t compute_capability = 0;
  DType input_dtype = DType::kFloat32;
  DType weight_dtype = DType::kFloat32;
  DType output_dtype = DType::kFloat32;
  MathMode math_mode = MathMode::kFp32Accumulate;
  LayoutId input_layout = LayoutId::kRowMajorDense;
  LayoutId weight_layout = LayoutId::kRowMajorDense;
  LayoutId output_layout = LayoutId::kRowMajorDense;
  AliasMode alias_mode = AliasMode::kDisjoint;
  GraphCompatibility graph = GraphCompatibility::kNotGraphCaptured;
  uint8_t rank = 0;
  std::array<uint64_t, kMaxDimensions> dimensions{};
  uint32_t query_heads = 0;
  uint32_t kv_heads = 0;
  uint32_t head_dimension = 0;
  uint32_t sequence_bucket = 0;
  uint64_t workspace_limit_bytes = 0;
  uint32_t alignment_class = 1;
  bool causal = false;

  friend bool operator==(const KernelKey&, const KernelKey&) = default;
};

[[nodiscard]] absl::Status ValidateKernelKey(const KernelKey& key);
[[nodiscard]] std::vector<std::byte> SerializeKernelKey(const KernelKey& key);
[[nodiscard]] uint64_t StableKernelKeyHash(const KernelKey& key) noexcept;
[[nodiscard]] bool KernelKeyLess(const KernelKey& left, const KernelKey& right);

}  // namespace inferx::ops

#endif  // INFERX_OPS_KERNEL_KEY_H_
