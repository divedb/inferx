#include "inferx/platform/cuda/ops/cuda_attention.h"

#include <cstddef>
#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"
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
                 "cuda_attention.tensor_interval.begin");
  if (!begin.ok()) return begin.status();
  return TensorInterval{tensor.buffer().allocation_id(), *begin,
                        tensor.layout().reachable_bytes.value()};
}

bool Overlaps(const TensorInterval& left, const TensorInterval& right) noexcept {
  if (left.allocation != right.allocation || left.size == 0 || right.size == 0) return false;
  if (left.begin <= right.begin) return right.begin - left.begin < left.size;
  return left.begin - right.begin < right.size;
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

absl::Status ValidateIndptr(std::span<const int32_t> indptr, uint64_t expected_end) {
  if (indptr.empty() || indptr.front() != 0 || indptr.back() < 0 ||
      static_cast<uint64_t>(indptr.back()) != expected_end) {
    return absl::InvalidArgumentError("cuda_attention.indptr: invalid endpoints");
  }
  for (size_t index = 1; index < indptr.size(); ++index) {
    if (indptr[index] < indptr[index - 1]) {
      return absl::InvalidArgumentError("cuda_attention.indptr: offsets must be nondecreasing");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateRequest(const inferx::ops::AttentionRequest& request,
                             const DeviceAttentionMetadata& metadata,
                             const CudaOpContext& context) {
  if (request.phase != inferx::ops::ExecutionPhase::kPrefill &&
      request.phase != inferx::ops::ExecutionPhase::kDecode) {
    return absl::InvalidArgumentError("cuda_attention.phase: expected prefill or decode");
  }
  if (!SupportedStorage(request.query.dtype()) ||
      request.query.dtype() != request.new_key.dtype() ||
      request.query.dtype() != request.new_value.dtype() ||
      request.query.dtype() != request.key_cache.dtype() ||
      request.query.dtype() != request.value_cache.dtype() ||
      request.query.dtype() != request.output.dtype()) {
    return absl::UnimplementedError("cuda_attention.dtype: one FP32/FP16/BF16 family is required");
  }
  if (request.query.shape().rank() != 3 || request.new_key.shape().rank() != 3 ||
      request.new_value.shape().rank() != 3 || request.key_cache.shape().rank() != 4 ||
      request.value_cache.shape().rank() != 4 || request.output.shape().rank() != 3 ||
      !request.query.layout().contiguous || !request.new_key.layout().contiguous ||
      !request.new_value.layout().contiguous || !request.key_cache.layout().contiguous ||
      !request.value_cache.layout().contiguous || !request.output.layout().contiguous) {
    return absl::UnimplementedError(
        "cuda_attention.layout: contiguous QKV and BSHD cache required");
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
    return absl::UnimplementedError("cuda_attention.shape: outside fallback capability envelope");
  }
  if (request.phase == inferx::ops::ExecutionPhase::kPrefill && total_queries > 512) {
    return absl::UnimplementedError(
        "cuda_attention.prefill: total query tokens exceed fallback envelope");
  }
  if (request.query_indptr.size() != batch + 1 || request.new_kv_indptr.size() != batch + 1 ||
      request.query_positions.size() != total_queries ||
      request.kv_lengths_before.size() != batch) {
    return absl::InvalidArgumentError("cuda_attention.metadata: host metadata sizes mismatch");
  }
  absl::Status status = ValidateIndptr(request.query_indptr, total_queries);
  if (!status.ok()) return status;
  status = ValidateIndptr(request.new_kv_indptr, total_new);
  if (!status.ok()) return status;
  for (size_t sequence = 0; sequence < static_cast<size_t>(batch); ++sequence) {
    const int32_t query_count = request.query_indptr[sequence + 1] - request.query_indptr[sequence];
    const int32_t new_count = request.new_kv_indptr[sequence + 1] - request.new_kv_indptr[sequence];
    const int32_t before = request.kv_lengths_before[sequence];
    if (before < 0 || query_count != new_count ||
        (request.phase == inferx::ops::ExecutionPhase::kDecode && query_count > 1) ||
        static_cast<uint64_t>(before) + static_cast<uint64_t>(new_count) > max_context) {
      return absl::InvalidArgumentError("cuda_attention.sequence: invalid counts or KV capacity");
    }
    const size_t begin = static_cast<size_t>(request.query_indptr[sequence]);
    for (int32_t local = 0; local < query_count; ++local) {
      const int64_t expected = static_cast<int64_t>(before) + static_cast<int64_t>(local);
      if (static_cast<int64_t>(request.query_positions[begin + static_cast<size_t>(local)]) !=
          expected) {
        return absl::InvalidArgumentError(
            "cuda_attention.positions: expected contiguous positions");
      }
    }
  }
  if ((total_queries != 0 &&
       (metadata.query_indptr == nullptr || metadata.query_positions == nullptr)) ||
      (total_new != 0 &&
       (metadata.new_kv_indptr == nullptr || metadata.kv_lengths_before == nullptr))) {
    return absl::InvalidArgumentError("cuda_attention.metadata: device metadata is null");
  }
  for (const Device device :
       {request.query.buffer().device(), request.new_key.buffer().device(),
        request.new_value.buffer().device(), request.key_cache.buffer().device(),
        request.value_cache.buffer().device(), request.output.buffer().device()}) {
    if (device.kind != DeviceKind::kCuda || device.ordinal != context.device) {
      return absl::InvalidArgumentError("cuda_attention.device: operand/context mismatch");
    }
  }
  if (request.query.buffer().memory_kind() != MemoryKind::kDevice ||
      request.new_key.buffer().memory_kind() != MemoryKind::kDevice ||
      request.new_value.buffer().memory_kind() != MemoryKind::kDevice ||
      request.key_cache.buffer().memory_kind() != MemoryKind::kDevice ||
      request.value_cache.buffer().memory_kind() != MemoryKind::kDevice ||
      request.output.buffer().memory_kind() != MemoryKind::kDevice) {
    return absl::InvalidArgumentError("cuda_attention.memory: device memory is required");
  }
  if (context.stream.device() != context.device) {
    return absl::InvalidArgumentError("cuda_attention.stream: stream/context mismatch");
  }
  status = RejectOverlap(request.query, request.new_key, "cuda_attention.query_new_key_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.new_value, "cuda_attention.query_new_value_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_key, request.new_value, "cuda_attention.new_kv_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.key_cache, "cuda_attention.query_cache_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.value_cache, "cuda_attention.query_cache_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_key, request.key_cache, "cuda_attention.new_key_cache_alias");
  if (!status.ok()) return status;
  status =
      RejectOverlap(request.new_key, request.value_cache, "cuda_attention.new_key_cache_alias");
  if (!status.ok()) return status;
  status =
      RejectOverlap(request.new_value, request.key_cache, "cuda_attention.new_value_cache_alias");
  if (!status.ok()) return status;
  status =
      RejectOverlap(request.new_value, request.value_cache, "cuda_attention.new_value_cache_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.key_cache, request.value_cache, "cuda_attention.cache_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.output, "cuda_attention.output_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_key, request.output, "cuda_attention.output_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_value, request.output, "cuda_attention.output_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.key_cache, request.output, "cuda_attention.output_alias");
  if (!status.ok()) return status;
  status = RejectOverlap(request.value_cache, request.output, "cuda_attention.output_alias");
  if (!status.ok()) return status;
  return context.health == nullptr ? absl::OkStatus() : context.health->CheckAcceptingWork();
}

}  // namespace

absl::Status LaunchAttentionFallback(const inferx::ops::AttentionRequest& request,
                                     const DeviceAttentionMetadata& metadata,
                                     const CudaOpContext& context) {
  absl::Status status = ValidateRequest(request, metadata, context);
  if (!status.ok()) return status;
  const uint64_t total_queries = request.query.shape().dim(0);
  const uint64_t total_new = request.new_key.shape().dim(0);
  if (total_queries == 0) return absl::OkStatus();
  const uint64_t batch = request.key_cache.shape().dim(0);
  const uint64_t max_context = request.key_cache.shape().dim(1);
  const uint64_t query_heads = request.query.shape().dim(1);
  const uint64_t kv_heads = request.new_key.shape().dim(1);
  const uint64_t head_dimension = request.query.shape().dim(2);
  status = CheckCuda(
      kernels::LaunchKvAppendKernel(
          Address(request.new_key), Address(request.new_value), Address(request.key_cache),
          Address(request.value_cache), metadata.new_kv_indptr, metadata.kv_lengths_before, batch,
          total_new, max_context, kv_heads, head_dimension,
          ToKernelStorageType(request.query.dtype()), context.stream.handle()),
      "ops.kv_append.launch", context.device, context.health);
  if (!status.ok()) return status;
  return CheckCuda(
      kernels::LaunchAttentionKernel(
          Address(request.query), Address(request.key_cache), Address(request.value_cache),
          Address(request.output), metadata.query_indptr, metadata.query_positions, batch,
          total_queries, max_context, query_heads, kv_heads, head_dimension,
          ToKernelStorageType(request.query.dtype()), context.stream.handle()),
      "ops.attention_fallback.launch", context.device, context.health);
}

}  // namespace inferx::cuda::ops
