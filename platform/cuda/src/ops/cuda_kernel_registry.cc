#include "inferx/platform/cuda/ops/cuda_kernel_registry.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"
#include "inferx/ops/backend_capability.h"
#include "inferx/platform/cuda/cuda_buffer_access.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "ops_kernels.h"

namespace inferx::cuda::ops {
namespace {

struct TensorInterval {
  AllocationId allocation;
  uint64_t begin = 0;
  uint64_t size = 0;
};

template <typename View>
absl::StatusOr<TensorInterval> Interval(const View& tensor) {
  absl::StatusOr<uint64_t> begin =
      CheckedAdd(tensor.buffer().range().offset.value(), tensor.byte_offset().value(),
                 "cuda_ops.tensor_interval.begin");
  if (!begin.ok()) return begin.status();
  return TensorInterval{tensor.buffer().allocation_id(), *begin,
                        tensor.layout().reachable_bytes.value()};
}

bool Overlaps(const TensorInterval& left, const TensorInterval& right) noexcept {
  if (left.allocation != right.allocation || left.size == 0 || right.size == 0) return false;
  if (left.begin <= right.begin) return right.begin - left.begin < left.size;
  return left.begin - right.begin < right.size;
}

bool ExactAlias(const TensorInterval& left, const TensorInterval& right) noexcept {
  return left.allocation == right.allocation && left.begin == right.begin &&
         left.size == right.size;
}

template <typename First, typename Second>
absl::Status RejectOverlap(const First& first, const Second& second, const char* field) {
  absl::StatusOr<TensorInterval> first_interval = Interval(first);
  if (!first_interval.ok()) return first_interval.status();
  absl::StatusOr<TensorInterval> second_interval = Interval(second);
  if (!second_interval.ok()) return second_interval.status();
  if (Overlaps(*first_interval, *second_interval)) {
    return absl::InvalidArgumentError(absl::StrCat(field, ": tensor ranges overlap"));
  }
  return absl::OkStatus();
}

template <typename Input>
absl::Status RequireExactOrDisjoint(const Input& input, const MutableTensorView& output,
                                    const char* field) {
  absl::StatusOr<TensorInterval> input_interval = Interval(input);
  if (!input_interval.ok()) return input_interval.status();
  absl::StatusOr<TensorInterval> output_interval = Interval(output);
  if (!output_interval.ok()) return output_interval.status();
  if (Overlaps(*input_interval, *output_interval) &&
      !ExactAlias(*input_interval, *output_interval)) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, ": only an exact in-place alias is allowed"));
  }
  return absl::OkStatus();
}

absl::Status ValidateBinaryAlias(const TensorView& left, const TensorView& right,
                                 const MutableTensorView& output, const char* field,
                                 bool allow_both_exact) {
  absl::StatusOr<TensorInterval> left_interval = Interval(left);
  if (!left_interval.ok()) return left_interval.status();
  absl::StatusOr<TensorInterval> right_interval = Interval(right);
  if (!right_interval.ok()) return right_interval.status();
  absl::StatusOr<TensorInterval> output_interval = Interval(output);
  if (!output_interval.ok()) return output_interval.status();
  const bool exact_left = ExactAlias(*left_interval, *output_interval);
  const bool exact_right = ExactAlias(*right_interval, *output_interval);
  if ((Overlaps(*left_interval, *output_interval) && !exact_left) ||
      (Overlaps(*right_interval, *output_interval) && !exact_right) ||
      (!allow_both_exact && exact_left && Overlaps(*right_interval, *output_interval)) ||
      (!allow_both_exact && exact_right && Overlaps(*left_interval, *output_interval))) {
    return absl::InvalidArgumentError(absl::StrCat(field, ": unsupported tensor overlap"));
  }
  return absl::OkStatus();
}

const void* Address(const TensorView& tensor) noexcept {
  const auto* base = static_cast<const std::byte*>(BufferAccess::Address(tensor.buffer()));
  return base == nullptr ? nullptr : base + tensor.byte_offset().value();
}

void* Address(const MutableTensorView& tensor) noexcept {
  auto* base = static_cast<std::byte*>(BufferAccess::Address(tensor.buffer()));
  return base == nullptr ? nullptr : base + tensor.byte_offset().value();
}

bool SupportedStorage(DType dtype) noexcept {
  return dtype == DType::kFloat32 || dtype == DType::kFloat16 || dtype == DType::kBFloat16;
}

kernels::StorageType ToKernelStorageType(DType dtype) noexcept {
  if (dtype == DType::kFloat16) return kernels::StorageType::kFloat16;
  if (dtype == DType::kBFloat16) return kernels::StorageType::kBFloat16;
  return kernels::StorageType::kFloat32;
}

template <typename View>
absl::Status ValidateCudaTensor(const View& tensor, uint8_t rank, const char* field) {
  if (!SupportedStorage(tensor.dtype())) {
    return absl::UnimplementedError(std::string(field) + ": FP32, FP16, or BF16 is required");
  }
  if (tensor.shape().rank() != rank || !tensor.layout().contiguous) {
    return absl::UnimplementedError(std::string(field) + ": contiguous rank mismatch");
  }
  if (tensor.buffer().device().kind != DeviceKind::kCuda ||
      tensor.buffer().memory_kind() != MemoryKind::kDevice) {
    return absl::InvalidArgumentError(std::string(field) + ": CUDA device memory is required");
  }
  return absl::OkStatus();
}

template <typename View>
absl::Status ValidateContextDevice(const View& tensor, const CudaOpContext& context) {
  if (tensor.buffer().device().ordinal != context.device ||
      context.stream.device() != context.device) {
    return absl::InvalidArgumentError("cuda_ops.device: tensor, stream, and context differ");
  }
  return context.health == nullptr ? absl::OkStatus() : context.health->CheckAcceptingWork();
}

uint32_t StorageMask() noexcept {
  return inferx::ops::DTypeMask(DType::kFloat32) | inferx::ops::DTypeMask(DType::kFloat16) |
         inferx::ops::DTypeMask(DType::kBFloat16);
}

inferx::ops::BackendCapability GlueCapability(
    inferx::ops::OpKind op, std::string id, uint8_t rank, uint16_t compute_capability,
    inferx::ops::LayoutId layout = inferx::ops::LayoutId::kRowMajorDense) {
  inferx::ops::BackendCapability capability;
  capability.backend = inferx::ops::BackendId::kInferxCuda;
  capability.capability_id = std::move(id);
  capability.priority = 100;
  capability.op = op;
  capability.device_kind = DeviceKind::kCuda;
  capability.minimum_compute_capability = compute_capability;
  capability.maximum_compute_capability = compute_capability;
  capability.input_dtype_mask = StorageMask();
  capability.weight_dtype_mask = StorageMask();
  capability.output_dtype_mask = StorageMask();
  capability.require_matching_input_weight_dtype = true;
  capability.require_matching_weight_output_dtype = true;
  capability.input_layout = layout;
  capability.weight_layout = layout;
  capability.output_layout = layout;
  capability.alias_mask = inferx::ops::AliasMask(inferx::ops::AliasMode::kDisjoint);
  capability.rank = rank;
  for (size_t axis = 0; axis < rank; ++axis) {
    capability.dimensions[axis] = {0, static_cast<uint64_t>(std::numeric_limits<int32_t>::max())};
  }
  capability.minimum_operand_alignment = 2;
  capability.dependency_version = "inferx-owned-v1";
  return capability;
}

inferx::ops::BackendCapability CublasLtCapability(inferx::ops::OpKind op, std::string id,
                                                  uint16_t compute_capability) {
  inferx::ops::BackendCapability capability =
      GlueCapability(op, std::move(id), 3, compute_capability);
  capability.backend = inferx::ops::BackendId::kCublasLt;
  capability.priority = 50;
  capability.preparation_uses_heuristics = true;
  capability.require_matching_weight_output_dtype = false;
  capability.workspace_alignment = 256;
  capability.dependency_version = "cuda-13-cublaslt";
  return capability;
}

}  // namespace

absl::Status RegisterCudaCapabilities(inferx::ops::KernelRegistry& registry,
                                      uint16_t compute_capability) {
  absl::Status status = registry.Register(
      CublasLtCapability(inferx::ops::OpKind::kGemm, "cublaslt.gemm.row_major_fp32_accumulate.v1",
                         compute_capability));
  if (!status.ok()) return status;
  auto logits = CublasLtCapability(inferx::ops::OpKind::kLogits,
                                   "cublaslt.logits.row_major_fp32.v1", compute_capability);
  logits.output_dtype_mask = inferx::ops::DTypeMask(DType::kFloat32);
  status = registry.Register(std::move(logits));
  if (!status.ok()) return status;

  auto embedding = GlueCapability(inferx::ops::OpKind::kEmbedding, "inferx_cuda.embedding.v1", 2,
                                  compute_capability);
  embedding.input_dtype_mask = inferx::ops::DTypeMask(DType::kInt32);
  embedding.require_matching_input_weight_dtype = false;
  status = registry.Register(std::move(embedding));
  if (!status.ok()) return status;
  status = registry.Register(GlueCapability(inferx::ops::OpKind::kRmsNorm,
                                            "inferx_cuda.rms_norm.v1", 2, compute_capability));
  if (!status.ok()) return status;
  auto rope = GlueCapability(inferx::ops::OpKind::kRope, "inferx_cuda.rope.v1", 4,
                             compute_capability, inferx::ops::LayoutId::kQkvTokenHeadDim);
  rope.weight_layout = inferx::ops::LayoutId::kRowMajorDense;
  rope.alias_mask = static_cast<uint8_t>(
      rope.alias_mask | inferx::ops::AliasMask(inferx::ops::AliasMode::kExactInPlace));
  status = registry.Register(std::move(rope));
  if (!status.ok()) return status;
  auto swiglu = GlueCapability(inferx::ops::OpKind::kSiluMultiply, "inferx_cuda.silu_multiply.v1",
                               2, compute_capability);
  swiglu.alias_mask = static_cast<uint8_t>(
      swiglu.alias_mask | inferx::ops::AliasMask(inferx::ops::AliasMode::kExactLeft) |
      inferx::ops::AliasMask(inferx::ops::AliasMode::kExactRight));
  status = registry.Register(std::move(swiglu));
  if (!status.ok()) return status;
  auto residual = GlueCapability(inferx::ops::OpKind::kResidual, "inferx_cuda.residual.v1", 2,
                                 compute_capability);
  residual.alias_mask = static_cast<uint8_t>(
      residual.alias_mask | inferx::ops::AliasMask(inferx::ops::AliasMode::kExactLeft) |
      inferx::ops::AliasMask(inferx::ops::AliasMode::kExactRight));
  status = registry.Register(std::move(residual));
  if (!status.ok()) return status;
  auto attention =
      GlueCapability(inferx::ops::OpKind::kAttention, "inferx_cuda.attention_fallback.v1", 4,
                     compute_capability, inferx::ops::LayoutId::kQkvTokenHeadDim);
  attention.phase_mask = inferx::ops::PhaseMask(inferx::ops::ExecutionPhase::kPrefill) |
                         inferx::ops::PhaseMask(inferx::ops::ExecutionPhase::kDecode);
  attention.weight_layout = inferx::ops::LayoutId::kContiguousKvBshd;
  attention.maximum_query_heads = 64;
  attention.maximum_kv_heads = 64;
  attention.maximum_head_dimension = 256;
  attention.head_dimension_multiple = 2;
  attention.maximum_sequence_bucket = 512;
  attention.dimensions[0] = {1, 8};
  attention.dimensions[1] = {1, 4096};
  attention.dimensions[2] = {1, 64};
  attention.dimensions[3] = {2, 256};
  return registry.Register(std::move(attention));
}

absl::Status LaunchEmbedding(const inferx::ops::EmbeddingRequest& request,
                             const CudaOpContext& context) {
  if (request.token_ids.dtype() != DType::kInt32 || request.token_ids.shape().rank() != 1 ||
      !request.token_ids.layout().contiguous ||
      request.token_ids.buffer().device().kind != DeviceKind::kCuda ||
      request.token_ids.buffer().memory_kind() != MemoryKind::kDevice) {
    return absl::InvalidArgumentError("cuda_embedding.token_ids: contiguous INT32 IDs required");
  }
  absl::Status status = ValidateCudaTensor(request.weight, 2, "cuda_embedding.weight");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.output, 2, "cuda_embedding.output");
  if (!status.ok()) return status;
  if (request.weight.dtype() != request.output.dtype() || request.weight.shape().dim(0) == 0 ||
      request.weight.shape().dim(1) == 0 ||
      request.output.shape().dim(0) != request.token_ids.shape().dim(0) ||
      request.output.shape().dim(1) != request.weight.shape().dim(1)) {
    return absl::InvalidArgumentError("cuda_embedding.shape: incompatible tensors");
  }
  if (request.host_token_ids.size() != request.token_ids.shape().dim(0)) {
    return absl::InvalidArgumentError(
        "cuda_embedding.host_token_ids: an exact immutable host mirror is required");
  }
  for (int32_t token_id : request.host_token_ids) {
    if (token_id < 0 || static_cast<uint64_t>(token_id) >= request.weight.shape().dim(0)) {
      return absl::InvalidArgumentError("cuda_embedding.token_ids: ID is outside vocabulary");
    }
  }
  status = ValidateContextDevice(request.token_ids, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.weight, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.output, context);
  if (!status.ok()) return status;
  status = RejectOverlap(request.token_ids, request.weight, "cuda_embedding.alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.token_ids, request.output, "cuda_embedding.alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.weight, request.output, "cuda_embedding.alias");
  if (!status.ok()) return status;
  return CheckCuda(
      kernels::LaunchEmbeddingKernel(
          static_cast<const int32_t*>(Address(request.token_ids)), Address(request.weight),
          Address(request.output), request.token_ids.shape().dim(0), request.weight.shape().dim(0),
          request.weight.shape().dim(1), ToKernelStorageType(request.weight.dtype()),
          context.stream.handle()),
      "ops.embedding.launch", context.device, context.health);
}

absl::Status LaunchRmsNorm(const inferx::ops::RmsNormRequest& request,
                           const CudaOpContext& context) {
  absl::Status status = ValidateCudaTensor(request.input, 2, "cuda_rms_norm.input");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.weight, 1, "cuda_rms_norm.weight");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.output, 2, "cuda_rms_norm.output");
  if (!status.ok()) return status;
  if (request.input.dtype() != request.weight.dtype() ||
      request.input.dtype() != request.output.dtype() ||
      request.input.shape() != request.output.shape() || request.input.shape().dim(1) == 0 ||
      request.weight.shape().dim(0) != request.input.shape().dim(1) ||
      !std::isfinite(request.epsilon) || request.epsilon <= 0.0F) {
    return absl::InvalidArgumentError("cuda_rms_norm: incompatible shape, dtype, or epsilon");
  }
  status = ValidateContextDevice(request.input, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.weight, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.output, context);
  if (!status.ok()) return status;
  status = RejectOverlap(request.input, request.output, "cuda_rms_norm.alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.weight, request.output, "cuda_rms_norm.alias");
  if (!status.ok()) return status;
  return CheckCuda(kernels::LaunchRmsNormKernel(
                       Address(request.input), Address(request.weight), Address(request.output),
                       request.input.shape().dim(0), request.input.shape().dim(1), request.epsilon,
                       ToKernelStorageType(request.input.dtype()), context.stream.handle()),
                   "ops.rms_norm.launch", context.device, context.health);
}

absl::Status LaunchRope(const inferx::ops::RopeRequest& request, const CudaOpContext& context) {
  absl::Status status = ValidateCudaTensor(request.query, 3, "cuda_rope.query");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.key, 3, "cuda_rope.key");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.query_output, 3, "cuda_rope.query_output");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.key_output, 3, "cuda_rope.key_output");
  if (!status.ok()) return status;
  if (request.positions.dtype() != DType::kInt32 || request.positions.shape().rank() != 1 ||
      !request.positions.layout().contiguous ||
      request.positions.buffer().device().kind != DeviceKind::kCuda ||
      request.positions.buffer().memory_kind() != MemoryKind::kDevice ||
      request.query.dtype() != request.key.dtype() ||
      request.query.dtype() != request.query_output.dtype() ||
      request.query.dtype() != request.key_output.dtype() ||
      request.query_output.shape() != request.query.shape() ||
      request.key_output.shape() != request.key.shape() ||
      request.query.shape().dim(0) != request.key.shape().dim(0) ||
      request.query.shape().dim(2) != request.key.shape().dim(2) ||
      request.query.shape().dim(2) == 0 || request.query.shape().dim(2) % 2 != 0 ||
      request.positions.shape().dim(0) != request.query.shape().dim(0) ||
      !std::isfinite(request.theta) || request.theta <= 0.0F ||
      request.max_position_embeddings == 0) {
    return absl::InvalidArgumentError("cuda_rope: incompatible shape, dtype, or theta");
  }
  if (request.host_positions.size() != request.positions.shape().dim(0)) {
    return absl::InvalidArgumentError(
        "cuda_rope.host_positions: an exact immutable host mirror is required");
  }
  for (int32_t position : request.host_positions) {
    if (position < 0 || static_cast<uint64_t>(position) >= request.max_position_embeddings) {
      return absl::InvalidArgumentError("cuda_rope.positions: position is outside model range");
    }
  }
  status = ValidateContextDevice(request.query, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.key, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.positions, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.query_output, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.key_output, context);
  if (!status.ok()) return status;
  status = RequireExactOrDisjoint(request.query, request.query_output, "cuda_rope.query_alias");
  if (!status.ok()) return status;
  status = RequireExactOrDisjoint(request.key, request.key_output, "cuda_rope.key_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.key, "cuda_rope.query_key_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query_output, request.key, "cuda_rope.query_key_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.key_output, "cuda_rope.query_key_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query_output, request.key_output, "cuda_rope.query_key_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.positions, request.query_output, "cuda_rope.position_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.positions, request.key_output, "cuda_rope.position_alias");
  if (!status.ok()) return status;
  return CheckCuda(
      kernels::LaunchRopeKernel(
          Address(request.query), Address(request.key),
          static_cast<const int32_t*>(Address(request.positions)), Address(request.query_output),
          Address(request.key_output), request.query.shape().dim(0), request.query.shape().dim(1),
          request.key.shape().dim(1), request.query.shape().dim(2), request.theta,
          request.max_position_embeddings, ToKernelStorageType(request.query.dtype()),
          context.stream.handle()),
      "ops.rope.launch", context.device, context.health);
}

absl::Status LaunchSwiGlu(const inferx::ops::SwiGluRequest& request, const CudaOpContext& context) {
  absl::Status status = ValidateCudaTensor(request.left, 2, "cuda_swiglu.gate");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.right, 2, "cuda_swiglu.up");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.output, 2, "cuda_swiglu.output");
  if (!status.ok()) return status;
  if (request.left.shape() != request.right.shape() ||
      request.left.shape() != request.output.shape() ||
      request.left.dtype() != request.right.dtype() ||
      request.left.dtype() != request.output.dtype()) {
    return absl::InvalidArgumentError("cuda_swiglu: tensors must have equal shape and dtype");
  }
  status = ValidateContextDevice(request.left, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.right, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.output, context);
  if (!status.ok()) return status;
  status =
      ValidateBinaryAlias(request.left, request.right, request.output, "cuda_swiglu.alias", false);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> elements = request.left.shape().NumElements();
  if (!elements.ok()) return elements.status();
  return CheckCuda(kernels::LaunchSwiGluKernel(Address(request.left), Address(request.right),
                                               Address(request.output), *elements,
                                               ToKernelStorageType(request.left.dtype()),
                                               context.stream.handle()),
                   "ops.silu_multiply.launch", context.device, context.health);
}

absl::Status LaunchResidual(const inferx::ops::ResidualRequest& request,
                            const CudaOpContext& context) {
  absl::Status status = ValidateCudaTensor(request.left, 2, "cuda_residual.left");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.right, 2, "cuda_residual.right");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.output, 2, "cuda_residual.output");
  if (!status.ok()) return status;
  if (request.left.shape() != request.right.shape() ||
      request.left.shape() != request.output.shape() ||
      request.left.dtype() != request.right.dtype() ||
      request.left.dtype() != request.output.dtype()) {
    return absl::InvalidArgumentError("cuda_residual: tensors must have equal shape and dtype");
  }
  status = ValidateContextDevice(request.left, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.right, context);
  if (!status.ok()) return status;
  status = ValidateContextDevice(request.output, context);
  if (!status.ok()) return status;
  status =
      ValidateBinaryAlias(request.left, request.right, request.output, "cuda_residual.alias", true);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> elements = request.left.shape().NumElements();
  if (!elements.ok()) return elements.status();
  return CheckCuda(kernels::LaunchResidualKernel(Address(request.left), Address(request.right),
                                                 Address(request.output), *elements,
                                                 ToKernelStorageType(request.left.dtype()),
                                                 context.stream.handle()),
                   "ops.residual.launch", context.device, context.health);
}

}  // namespace inferx::cuda::ops
