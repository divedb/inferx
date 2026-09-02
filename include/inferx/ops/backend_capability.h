#ifndef INFERX_OPS_BACKEND_CAPABILITY_H_
#define INFERX_OPS_BACKEND_CAPABILITY_H_

#include <array>
#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "inferx/ops/kernel_key.h"

namespace inferx::ops {

struct DimensionRange {
  uint64_t minimum = 0;
  uint64_t maximum = 0;
};

struct CapabilityMatch {
  uint64_t workspace_bytes = 0;
  uint32_t workspace_alignment = 1;
};

struct BackendCapability {
  BackendId backend = BackendId::kReference;
  std::string capability_id;
  uint32_t priority = 0;
  OpKind op = OpKind::kEmbedding;
  uint8_t phase_mask = 0x07;
  DeviceKind device_kind = DeviceKind::kHost;
  uint16_t minimum_compute_capability = 0;
  uint16_t maximum_compute_capability = 0;
  uint32_t input_dtype_mask = 0;
  uint32_t weight_dtype_mask = 0;
  uint32_t output_dtype_mask = 0;
  bool require_matching_input_weight_dtype = false;
  bool require_matching_weight_output_dtype = false;
  LayoutId input_layout = LayoutId::kRowMajorDense;
  LayoutId weight_layout = LayoutId::kRowMajorDense;
  LayoutId output_layout = LayoutId::kRowMajorDense;
  uint8_t alias_mask = 0x01;
  uint8_t rank = 0;
  std::array<DimensionRange, KernelKey::kMaxDimensions> dimensions{};
  uint32_t maximum_query_heads = 0;
  uint32_t maximum_kv_heads = 0;
  uint32_t maximum_head_dimension = 0;
  uint32_t head_dimension_multiple = 1;
  uint32_t maximum_sequence_bucket = 0;
  uint32_t minimum_operand_alignment = 1;
  uint64_t fixed_workspace_bytes = 0;
  uint64_t workspace_bytes_per_output_element = 0;
  uint32_t workspace_alignment = 1;
  bool deterministic = true;
  bool preparation_uses_heuristics = false;
  bool launch_allocates = false;
  std::string contract_version = "v1";
  std::string dependency_version;

  [[nodiscard]] absl::Status Validate() const;
  [[nodiscard]] absl::StatusOr<CapabilityMatch> Match(const KernelKey& key) const;
};

[[nodiscard]] constexpr uint32_t DtypeMask(Dtype dtype) noexcept {
  return uint32_t{1} << static_cast<uint8_t>(dtype);
}
[[nodiscard]] constexpr uint8_t PhaseMask(ExecutionPhase phase) noexcept {
  return static_cast<uint8_t>(uint8_t{1} << static_cast<uint8_t>(phase));
}
[[nodiscard]] constexpr uint8_t AliasMask(AliasMode mode) noexcept {
  return static_cast<uint8_t>(uint8_t{1} << static_cast<uint8_t>(mode));
}

}  // namespace inferx::ops

#endif  // INFERX_OPS_BACKEND_CAPABILITY_H_
