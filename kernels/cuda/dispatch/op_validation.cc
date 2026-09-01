#include "op_validation.h"

#include <cmath>
#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "cuda_tensor_checks.h"

namespace inferx::kernels::cuda {

absl::Status ValidateEmbeddingForCuda(const ops::EmbeddingRequest& request,
                                      const CudaLaunchContext& context) {
  if (request.token_ids.dtype() != DType::kInt32 || request.token_ids.shape().rank() != 1 ||
      !request.token_ids.layout().contiguous ||
      request.token_ids.buffer().device().kind != DeviceKind::kCuda ||
      request.token_ids.buffer().memory_kind() != MemoryKind::kDevice) {
    return absl::InvalidArgumentError(
        "kernels_cuda.embedding.token_ids: contiguous INT32 IDs required");
  }
  absl::Status status = ValidateCudaTensor(request.weight, 2, "kernels_cuda.embedding.weight");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.output, 2, "kernels_cuda.embedding.output");
  if (!status.ok()) return status;
  if (request.weight.dtype() != request.output.dtype() || request.weight.shape().dim(0) == 0 ||
      request.weight.shape().dim(1) == 0 ||
      request.output.shape().dim(0) != request.token_ids.shape().dim(0) ||
      request.output.shape().dim(1) != request.weight.shape().dim(1)) {
    return absl::InvalidArgumentError("kernels_cuda.embedding.shape: incompatible tensors");
  }
  if (request.host_token_ids.size() != request.token_ids.shape().dim(0)) {
    return absl::InvalidArgumentError(
        "kernels_cuda.embedding.host_token_ids: an exact immutable host mirror is required");
  }
  for (int32_t token_id : request.host_token_ids) {
    if (token_id < 0 || static_cast<uint64_t>(token_id) >= request.weight.shape().dim(0)) {
      return absl::InvalidArgumentError(
          "kernels_cuda.embedding.token_ids: ID is outside vocabulary");
    }
  }
  status = ValidateDeviceMatch(request.token_ids, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.weight, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.output, context.device);
  if (!status.ok()) return status;
  status = RejectOverlap(request.token_ids, request.weight, "kernels_cuda.embedding.alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.token_ids, request.output, "kernels_cuda.embedding.alias");
  if (!status.ok()) return status;
  return RejectOverlap(request.weight, request.output, "kernels_cuda.embedding.alias");
}

absl::Status ValidateRmsNormForCuda(const ops::RmsNormRequest& request,
                                    const CudaLaunchContext& context) {
  absl::Status status = ValidateCudaTensor(request.input, 2, "kernels_cuda.rms_norm.input");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.weight, 1, "kernels_cuda.rms_norm.weight");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.output, 2, "kernels_cuda.rms_norm.output");
  if (!status.ok()) return status;
  if (request.input.dtype() != request.weight.dtype() ||
      request.input.dtype() != request.output.dtype() ||
      request.input.shape() != request.output.shape() || request.input.shape().dim(1) == 0 ||
      request.weight.shape().dim(0) != request.input.shape().dim(1) ||
      !std::isfinite(request.epsilon) || request.epsilon <= 0.0F) {
    return absl::InvalidArgumentError("kernels_cuda.rms_norm: incompatible shape, dtype, or epsilon");
  }
  status = ValidateDeviceMatch(request.input, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.weight, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.output, context.device);
  if (!status.ok()) return status;
  status = RejectOverlap(request.input, request.output, "kernels_cuda.rms_norm.alias");
  if (!status.ok()) return status;
  return RejectOverlap(request.weight, request.output, "kernels_cuda.rms_norm.alias");
}

absl::Status ValidateRopeForCuda(const ops::RopeRequest& request,
                                 const CudaLaunchContext& context) {
  absl::Status status = ValidateCudaTensor(request.query, 3, "kernels_cuda.rope.query");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.key, 3, "kernels_cuda.rope.key");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.query_output, 3, "kernels_cuda.rope.query_output");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.key_output, 3, "kernels_cuda.rope.key_output");
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
    return absl::InvalidArgumentError("kernels_cuda.rope: incompatible shape, dtype, or theta");
  }
  if (request.host_positions.size() != request.positions.shape().dim(0)) {
    return absl::InvalidArgumentError(
        "kernels_cuda.rope.host_positions: an exact immutable host mirror is required");
  }
  for (int32_t position : request.host_positions) {
    if (position < 0 || static_cast<uint64_t>(position) >= request.max_position_embeddings) {
      return absl::InvalidArgumentError(
          "kernels_cuda.rope.positions: position is outside model range");
    }
  }
  status = ValidateDeviceMatch(request.query, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.key, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.positions, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.query_output, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.key_output, context.device);
  if (!status.ok()) return status;
  status =
      RequireExactOrDisjoint(request.query, request.query_output, "kernels_cuda.rope.query_alias");
  if (!status.ok()) return status;
  status = RequireExactOrDisjoint(request.key, request.key_output, "kernels_cuda.rope.key_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.key, "kernels_cuda.rope.query_key_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query_output, request.key, "kernels_cuda.rope.query_key_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.key_output, "kernels_cuda.rope.query_key_alias");
  if (!status.ok()) return status;
  status =
      RejectOverlap(request.query_output, request.key_output, "kernels_cuda.rope.query_key_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.positions, request.query_output,
                         "kernels_cuda.rope.position_alias");
  if (!status.ok()) return status;
  return RejectOverlap(request.positions, request.key_output, "kernels_cuda.rope.position_alias");
}

absl::Status ValidateSwiGluForCuda(const ops::SwiGluRequest& request,
                                   const CudaLaunchContext& context) {
  absl::Status status = ValidateCudaTensor(request.left, 2, "kernels_cuda.swiglu.gate");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.right, 2, "kernels_cuda.swiglu.up");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.output, 2, "kernels_cuda.swiglu.output");
  if (!status.ok()) return status;
  if (request.left.shape() != request.right.shape() ||
      request.left.shape() != request.output.shape() ||
      request.left.dtype() != request.right.dtype() ||
      request.left.dtype() != request.output.dtype()) {
    return absl::InvalidArgumentError(
        "kernels_cuda.swiglu: tensors must have equal shape and dtype");
  }
  status = ValidateDeviceMatch(request.left, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.right, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.output, context.device);
  if (!status.ok()) return status;
  return ValidateBinaryAlias(request.left, request.right, request.output,
                             "kernels_cuda.swiglu.alias", /*allow_both_exact=*/false);
}

absl::Status ValidateResidualForCuda(const ops::ResidualRequest& request,
                                     const CudaLaunchContext& context) {
  absl::Status status = ValidateCudaTensor(request.left, 2, "kernels_cuda.residual.left");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.right, 2, "kernels_cuda.residual.right");
  if (!status.ok()) return status;
  status = ValidateCudaTensor(request.output, 2, "kernels_cuda.residual.output");
  if (!status.ok()) return status;
  if (request.left.shape() != request.right.shape() ||
      request.left.shape() != request.output.shape() ||
      request.left.dtype() != request.right.dtype() ||
      request.left.dtype() != request.output.dtype()) {
    return absl::InvalidArgumentError(
        "kernels_cuda.residual: tensors must have equal shape and dtype");
  }
  status = ValidateDeviceMatch(request.left, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.right, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(request.output, context.device);
  if (!status.ok()) return status;
  return ValidateBinaryAlias(request.left, request.right, request.output,
                             "kernels_cuda.residual.alias", /*allow_both_exact=*/true);
}

namespace {

absl::Status ValidateDenseProjection(const TensorView& input, const TensorView& weight,
                                     const std::optional<TensorView>& addend,
                                     const MutableTensorView& output, const char* field,
                                     const CudaLaunchContext& context) {
  absl::Status status = ValidateCudaTensor(input, 2, field);
  if (!status.ok()) return status;
  status = ValidateCudaTensor(weight, 2, field);
  if (!status.ok()) return status;
  status = ValidateCudaTensor(output, 2, field);
  if (!status.ok()) return status;
  if (addend.has_value()) {
    status = ValidateCudaTensor(*addend, 2, field);
    if (!status.ok()) return status;
    if (addend->shape() != output.shape() || addend->dtype() != output.dtype()) {
      return absl::InvalidArgumentError(
          absl::StrCat(field, ": addend must match output shape and dtype"));
    }
  }
  if (input.dtype() != weight.dtype() || input.dtype() != output.dtype() ||
      input.shape().dim(0) != output.shape().dim(0) ||
      input.shape().dim(1) != weight.shape().dim(1) ||
      weight.shape().dim(0) != output.shape().dim(1) || input.shape().dim(1) == 0 ||
      weight.shape().dim(0) == 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, ": input[tokens,in] x weight[out,in] -> output[tokens,out] required"));
  }
  status = ValidateDeviceMatch(input, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(weight, context.device);
  if (!status.ok()) return status;
  status = ValidateDeviceMatch(output, context.device);
  if (!status.ok()) return status;
  if (addend.has_value()) {
    status = ValidateDeviceMatch(*addend, context.device);
    if (!status.ok()) return status;
  }
  status = RejectOverlap(input, weight, field);
  if (!status.ok()) return status;
  status = RejectOverlap(input, output, field);
  if (!status.ok()) return status;
  status = RejectOverlap(weight, output, field);
  if (!status.ok()) return status;
  if (addend.has_value()) {
    status = RequireExactOrDisjoint(*addend, output, field);
    if (!status.ok()) return status;
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateGemmForCuda(const ops::GemmRequest& request,
                                 const CudaLaunchContext& context) {
  if (!std::isfinite(request.alpha) || !std::isfinite(request.beta)) {
    return absl::InvalidArgumentError("kernels_cuda.gemm: alpha and beta must be finite");
  }
  return ValidateDenseProjection(request.input, request.weight, request.addend, request.output,
                                 "kernels_cuda.gemm", context);
}

absl::Status ValidateLogitsForCuda(const ops::LogitsRequest& request,
                                   const CudaLaunchContext& context) {
  return ValidateDenseProjection(request.hidden, request.weight, std::nullopt, request.output,
                                 "kernels_cuda.logits", context);
}

absl::Status ValidateAttentionForCuda(const ops::AttentionRequest& request,
                                      const DeviceAttentionMetadata& metadata,
                                      const CudaLaunchContext& context) {
  if (request.phase != ops::ExecutionPhase::kPrefill &&
      request.phase != ops::ExecutionPhase::kDecode) {
    return absl::InvalidArgumentError("kernels_cuda.attention.phase: expected prefill or decode");
  }
  if (!SupportedStorage(request.query.dtype()) ||
      request.query.dtype() != request.new_key.dtype() ||
      request.query.dtype() != request.new_value.dtype() ||
      request.query.dtype() != request.key_cache.dtype() ||
      request.query.dtype() != request.value_cache.dtype() ||
      request.query.dtype() != request.output.dtype()) {
    return absl::UnimplementedError(
        "kernels_cuda.attention.dtype: one FP32/FP16/BF16 family is required");
  }
  if (request.query.shape().rank() != 3 || request.new_key.shape().rank() != 3 ||
      request.new_value.shape().rank() != 3 || request.key_cache.shape().rank() != 4 ||
      request.value_cache.shape().rank() != 4 || request.output.shape().rank() != 3 ||
      !request.query.layout().contiguous || !request.new_key.layout().contiguous ||
      !request.new_value.layout().contiguous || !request.key_cache.layout().contiguous ||
      !request.value_cache.layout().contiguous || !request.output.layout().contiguous) {
    return absl::UnimplementedError(
        "kernels_cuda.attention.layout: contiguous QKV and BSHD cache required");
  }
  const uint64_t total_queries = request.query.shape().dim(0);
  const uint64_t total_new = request.new_key.shape().dim(0);
  const uint64_t query_heads = request.query.shape().dim(1);
  const uint64_t kv_heads = request.new_key.shape().dim(1);
  const uint64_t head_dimension = request.query.shape().dim(2);
  const uint64_t batch = request.key_cache.shape().dim(0);
  const uint64_t max_context = request.key_cache.shape().dim(1);
  if (batch == 0 || batch > 8 || query_heads == 0 || query_heads > 64 || kv_heads == 0 ||
      kv_heads > query_heads || query_heads % kv_heads != 0 || head_dimension < 2 ||
      head_dimension > 256 || head_dimension % 2 != 0 || max_context == 0 || max_context > 4096 ||
      request.new_key.shape().dim(2) != head_dimension ||
      request.new_value.shape() != request.new_key.shape() ||
      request.key_cache.shape().dim(2) != kv_heads ||
      request.key_cache.shape().dim(3) != head_dimension ||
      request.value_cache.shape() != request.key_cache.shape() ||
      request.output.shape() != request.query.shape()) {
    return absl::UnimplementedError(
        "kernels_cuda.attention.shape: outside fallback capability envelope");
  }
  if (request.phase == ops::ExecutionPhase::kPrefill && total_queries > 512) {
    return absl::UnimplementedError(
        "kernels_cuda.attention.prefill: total query tokens exceed fallback envelope");
  }
  if (request.query_indptr.size() != batch + 1 || request.new_kv_indptr.size() != batch + 1 ||
      request.query_positions.size() != total_queries ||
      request.kv_lengths_before.size() != batch) {
    return absl::InvalidArgumentError(
        "kernels_cuda.attention.metadata: host metadata sizes mismatch");
  }
  absl::Status status = ValidateIndptr(request.query_indptr, total_queries);
  if (!status.ok()) return status;
  status = ValidateIndptr(request.new_kv_indptr, total_new);
  if (!status.ok()) return status;
  for (size_t sequence = 0; sequence < static_cast<size_t>(batch); ++sequence) {
    const int32_t query_count =
        request.query_indptr[sequence + 1] - request.query_indptr[sequence];
    const int32_t new_count =
        request.new_kv_indptr[sequence + 1] - request.new_kv_indptr[sequence];
    const int32_t before = request.kv_lengths_before[sequence];
    if (before < 0 || query_count != new_count ||
        (request.phase == ops::ExecutionPhase::kDecode && query_count > 1) ||
        static_cast<uint64_t>(before) + static_cast<uint64_t>(new_count) > max_context) {
      return absl::InvalidArgumentError(
          "kernels_cuda.attention.sequence: invalid counts or KV capacity");
    }
    const size_t begin = static_cast<size_t>(request.query_indptr[sequence]);
    for (int32_t local = 0; local < query_count; ++local) {
      const int64_t expected = static_cast<int64_t>(before) + static_cast<int64_t>(local);
      if (static_cast<int64_t>(request.query_positions[begin + static_cast<size_t>(local)]) !=
          expected) {
        return absl::InvalidArgumentError(
            "kernels_cuda.attention.positions: expected contiguous positions");
      }
    }
  }
  if ((total_queries != 0 &&
       (metadata.query_indptr == nullptr || metadata.query_positions == nullptr)) ||
      (total_new != 0 &&
       (metadata.new_kv_indptr == nullptr || metadata.kv_lengths_before == nullptr))) {
    return absl::InvalidArgumentError("kernels_cuda.attention.metadata: device metadata is null");
  }
  for (const Device device :
       {request.query.buffer().device(), request.new_key.buffer().device(),
        request.new_value.buffer().device(), request.key_cache.buffer().device(),
        request.value_cache.buffer().device(), request.output.buffer().device()}) {
    if (device.kind != DeviceKind::kCuda || device.ordinal != context.device) {
      return absl::InvalidArgumentError(
          "kernels_cuda.attention.device: operand/context mismatch");
    }
  }
  if (request.query.buffer().memory_kind() != MemoryKind::kDevice ||
      request.new_key.buffer().memory_kind() != MemoryKind::kDevice ||
      request.new_value.buffer().memory_kind() != MemoryKind::kDevice ||
      request.key_cache.buffer().memory_kind() != MemoryKind::kDevice ||
      request.value_cache.buffer().memory_kind() != MemoryKind::kDevice ||
      request.output.buffer().memory_kind() != MemoryKind::kDevice) {
    return absl::InvalidArgumentError("kernels_cuda.attention.memory: device memory is required");
  }
  status = RejectOverlap(request.query, request.new_key, "kernels_cuda.attention.query_new_key_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.new_value,
                         "kernels_cuda.attention.query_new_value_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_key, request.new_value, "kernels_cuda.attention.new_kv_alias");
  if (!status.ok()) return status;
  status =
      RejectOverlap(request.query, request.key_cache, "kernels_cuda.attention.query_cache_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.value_cache,
                         "kernels_cuda.attention.query_cache_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_key, request.key_cache,
                         "kernels_cuda.attention.new_key_cache_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_key, request.value_cache,
                         "kernels_cuda.attention.new_key_cache_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_value, request.key_cache,
                         "kernels_cuda.attention.new_value_cache_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_value, request.value_cache,
                         "kernels_cuda.attention.new_value_cache_alias");
  if (!status.ok()) return status;
  status =
      RejectOverlap(request.key_cache, request.value_cache, "kernels_cuda.attention.cache_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.output, "kernels_cuda.attention.output_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_key, request.output, "kernels_cuda.attention.output_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_value, request.output, "kernels_cuda.attention.output_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.key_cache, request.output, "kernels_cuda.attention.output_alias");
  if (!status.ok()) return status;
  return RejectOverlap(request.value_cache, request.output, "kernels_cuda.attention.output_alias");
}

}  // namespace inferx::kernels::cuda
