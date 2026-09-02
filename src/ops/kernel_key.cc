#include "inferx/ops/kernel_key.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/status/status.h"

namespace inferx::ops {
namespace {

template <typename T>
void AppendLittleEndian(std::vector<std::byte>& output, T value) {
  using Unsigned = std::make_unsigned_t<T>;
  Unsigned bits = static_cast<Unsigned>(value);
  for (size_t index = 0; index < sizeof(T); ++index) {
    output.push_back(static_cast<std::byte>((bits >> (index * 8U)) & Unsigned{0xff}));
  }
}

template <typename T>
void HashLittleEndian(uint64_t* hash, T value) noexcept {
  using Unsigned = std::make_unsigned_t<T>;
  Unsigned bits = static_cast<Unsigned>(value);
  for (size_t index = 0; index < sizeof(T); ++index) {
    *hash ^= static_cast<uint8_t>((bits >> (index * 8U)) & Unsigned{0xff});
    *hash *= 1'099'511'628'211ULL;
  }
}

}  // namespace

std::string_view OpKindName(OpKind value) noexcept {
  switch (value) {
    case OpKind::kEmbedding:
      return "embedding";
    case OpKind::kGemm:
      return "gemm";
    case OpKind::kRmsNorm:
      return "rms_norm";
    case OpKind::kRope:
      return "rope";
    case OpKind::kSilu:
      return "silu";
    case OpKind::kMultiply:
      return "multiply";
    case OpKind::kSiluMultiply:
      return "silu_multiply";
    case OpKind::kResidual:
      return "residual";
    case OpKind::kAttention:
      return "attention";
    case OpKind::kLogits:
      return "logits";
  }
  return "unknown";
}

std::string_view BackendIdName(BackendId value) noexcept {
  switch (value) {
    case BackendId::kReference:
      return "reference";
    case BackendId::kCublasLt:
      return "cublaslt";
    case BackendId::kInferxCuda:
      return "inferx_cuda";
    case BackendId::kFlashInfer:
      return "flashinfer";
    case BackendId::kCutlass:
      return "cutlass";
  }
  return "unknown";
}

absl::Status ValidateKernelKey(const KernelKey& key) {
  if (key.schema_version != KernelKey::kSchemaVersion) {
    return absl::InvalidArgumentError("kernel_key.schema_version: unsupported version");
  }
  if (key.rank > KernelKey::kMaxDimensions) {
    return absl::InvalidArgumentError("kernel_key.rank: exceeds durable key capacity");
  }
  if (key.op > OpKind::kLogits || key.phase > ExecutionPhase::kDecode ||
      key.device_kind > DeviceKind::kCuda || key.input_dtype > DType::kFloat64 ||
      key.weight_dtype > DType::kFloat64 || key.output_dtype > DType::kFloat64 ||
      key.math_mode > MathMode::kFp32Accumulate || key.input_layout > LayoutId::kContiguousKvBshd ||
      key.weight_layout > LayoutId::kContiguousKvBshd ||
      key.output_layout > LayoutId::kContiguousKvBshd ||
      key.alias_mode > AliasMode::kExactInPlace ||
      key.graph > GraphCompatibility::kNotGraphCaptured) {
    return absl::InvalidArgumentError("kernel_key.enum: value is outside closed vocabulary");
  }
  for (size_t axis = key.rank; axis < KernelKey::kMaxDimensions; ++axis) {
    if (key.dimensions[axis] != 0) {
      return absl::InvalidArgumentError("kernel_key.dimensions: trailing dimensions must be zero");
    }
  }
  if (key.alignment_class == 0 || (key.alignment_class & (key.alignment_class - 1U)) != 0) {
    return absl::InvalidArgumentError("kernel_key.alignment: must be a nonzero power of two");
  }
  if ((key.device_kind == DeviceKind::kHost && key.compute_capability != 0) ||
      (key.device_kind == DeviceKind::kCuda && key.compute_capability == 0)) {
    return absl::InvalidArgumentError(
        "kernel_key.compute_capability: host uses zero and CUDA uses a positive value");
  }
  if (key.op == OpKind::kAttention &&
      (key.query_heads == 0 || key.kv_heads == 0 || key.head_dimension == 0 ||
       key.query_heads % key.kv_heads != 0 || !key.causal ||
       key.phase == ExecutionPhase::kGeneric || key.rank != 4 ||
       key.dimensions[2] != key.query_heads || key.dimensions[3] != key.head_dimension)) {
    return absl::InvalidArgumentError("kernel_key.attention: invalid causal head geometry");
  }
  if (key.op != OpKind::kAttention &&
      (key.query_heads != 0 || key.kv_heads != 0 || key.head_dimension != 0 ||
       key.sequence_bucket != 0 || key.causal)) {
    return absl::InvalidArgumentError(
        "kernel_key.attention: non-attention key carries attention fields");
  }
  return absl::OkStatus();
}

std::vector<std::byte> SerializeKernelKey(const KernelKey& key) {
  std::vector<std::byte> output;
  output.reserve(128);
  AppendLittleEndian(output, key.schema_version);
  AppendLittleEndian(output, static_cast<uint8_t>(key.op));
  AppendLittleEndian(output, static_cast<uint8_t>(key.phase));
  AppendLittleEndian(output, static_cast<uint8_t>(key.device_kind));
  AppendLittleEndian(output, key.compute_capability);
  AppendLittleEndian(output, static_cast<uint8_t>(key.input_dtype));
  AppendLittleEndian(output, static_cast<uint8_t>(key.weight_dtype));
  AppendLittleEndian(output, static_cast<uint8_t>(key.output_dtype));
  AppendLittleEndian(output, static_cast<uint8_t>(key.math_mode));
  AppendLittleEndian(output, static_cast<uint8_t>(key.input_layout));
  AppendLittleEndian(output, static_cast<uint8_t>(key.weight_layout));
  AppendLittleEndian(output, static_cast<uint8_t>(key.output_layout));
  AppendLittleEndian(output, static_cast<uint8_t>(key.alias_mode));
  AppendLittleEndian(output, static_cast<uint8_t>(key.graph));
  AppendLittleEndian(output, key.rank);
  for (uint64_t dimension : key.dimensions) AppendLittleEndian(output, dimension);
  AppendLittleEndian(output, key.query_heads);
  AppendLittleEndian(output, key.kv_heads);
  AppendLittleEndian(output, key.head_dimension);
  AppendLittleEndian(output, key.sequence_bucket);
  AppendLittleEndian(output, key.workspace_limit_bytes);
  AppendLittleEndian(output, key.alignment_class);
  AppendLittleEndian(output, static_cast<uint8_t>(key.causal ? 1 : 0));
  return output;
}

uint64_t StableKernelKeyHash(const KernelKey& key) noexcept {
  uint64_t hash = 14'695'981'039'346'656'037ULL;
  HashLittleEndian(&hash, key.schema_version);
  HashLittleEndian(&hash, static_cast<uint8_t>(key.op));
  HashLittleEndian(&hash, static_cast<uint8_t>(key.phase));
  HashLittleEndian(&hash, static_cast<uint8_t>(key.device_kind));
  HashLittleEndian(&hash, key.compute_capability);
  HashLittleEndian(&hash, static_cast<uint8_t>(key.input_dtype));
  HashLittleEndian(&hash, static_cast<uint8_t>(key.weight_dtype));
  HashLittleEndian(&hash, static_cast<uint8_t>(key.output_dtype));
  HashLittleEndian(&hash, static_cast<uint8_t>(key.math_mode));
  HashLittleEndian(&hash, static_cast<uint8_t>(key.input_layout));
  HashLittleEndian(&hash, static_cast<uint8_t>(key.weight_layout));
  HashLittleEndian(&hash, static_cast<uint8_t>(key.output_layout));
  HashLittleEndian(&hash, static_cast<uint8_t>(key.alias_mode));
  HashLittleEndian(&hash, static_cast<uint8_t>(key.graph));
  HashLittleEndian(&hash, key.rank);
  for (uint64_t dimension : key.dimensions) HashLittleEndian(&hash, dimension);
  HashLittleEndian(&hash, key.query_heads);
  HashLittleEndian(&hash, key.kv_heads);
  HashLittleEndian(&hash, key.head_dimension);
  HashLittleEndian(&hash, key.sequence_bucket);
  HashLittleEndian(&hash, key.workspace_limit_bytes);
  HashLittleEndian(&hash, key.alignment_class);
  HashLittleEndian(&hash, static_cast<uint8_t>(key.causal ? 1 : 0));
  return hash;
}

bool KernelKeyLess(const KernelKey& left, const KernelKey& right) {
  return std::tie(left.schema_version, left.op, left.phase, left.device_kind,
                  left.compute_capability, left.input_dtype, left.weight_dtype, left.output_dtype,
                  left.math_mode, left.input_layout, left.weight_layout, left.output_layout,
                  left.alias_mode, left.graph, left.rank, left.dimensions, left.query_heads,
                  left.kv_heads, left.head_dimension, left.sequence_bucket,
                  left.workspace_limit_bytes, left.alignment_class, left.causal) <
         std::tie(right.schema_version, right.op, right.phase, right.device_kind,
                  right.compute_capability, right.input_dtype, right.weight_dtype,
                  right.output_dtype, right.math_mode, right.input_layout, right.weight_layout,
                  right.output_layout, right.alias_mode, right.graph, right.rank, right.dimensions,
                  right.query_heads, right.kv_heads, right.head_dimension, right.sequence_bucket,
                  right.workspace_limit_bytes, right.alignment_class, right.causal);
}

}  // namespace inferx::ops
