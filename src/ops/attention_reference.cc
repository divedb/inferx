#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/ops/attention.h"
#include "src/ops/validation_internal.h"

namespace inferx::ops {
namespace {

absl::Status ValidateIndptr(std::span<const int32_t> indptr, uint64_t expected_end,
                            const char* field) {
  if (indptr.empty() || indptr.front() != 0) {
    return absl::InvalidArgumentError(field);
  }
  for (size_t index = 1; index < indptr.size(); ++index) {
    if (indptr[index] < indptr[index - 1]) return absl::InvalidArgumentError(field);
  }
  if (indptr.back() < 0 || static_cast<uint64_t>(indptr.back()) != expected_end) {
    return absl::InvalidArgumentError(field);
  }
  return absl::OkStatus();
}

template <typename First, typename Second>
absl::Status RejectOverlap(const First& first, const Second& second, const char* field) {
  absl::StatusOr<internal::TensorInterval> first_interval = internal::Interval(first);
  if (!first_interval.ok()) return first_interval.status();
  absl::StatusOr<internal::TensorInterval> second_interval = internal::Interval(second);
  if (!second_interval.ok()) return second_interval.status();
  if (internal::Overlaps(*first_interval, *second_interval)) {
    return absl::InvalidArgumentError(field);
  }
  return absl::OkStatus();
}

}  // namespace

uint64_t AttentionReferenceScratchElements(const AttentionRequest& request) noexcept {
  return request.key_cache.shape().rank() == 4 ? request.key_cache.shape().dim(1) : 0;
}

absl::Status ValidateAttention(const AttentionRequest& request) {
  if (request.phase != ExecutionPhase::kPrefill && request.phase != ExecutionPhase::kDecode) {
    return absl::InvalidArgumentError("attention.phase: expected prefill or decode");
  }
  absl::Status status =
      internal::ValidateTensor(request.query, Dtype::kFloat32, 3, "attention.query");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.new_key, Dtype::kFloat32, 3, "attention.new_key");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.new_value, Dtype::kFloat32, 3, "attention.new_value");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.key_cache, Dtype::kFloat32, 4, "attention.key_cache");
  if (!status.ok()) return status;
  status =
      internal::ValidateTensor(request.value_cache, Dtype::kFloat32, 4, "attention.value_cache");
  if (!status.ok()) return status;
  status = internal::ValidateTensor(request.output, Dtype::kFloat32, 3, "attention.output");
  if (!status.ok()) return status;

  const uint64_t total_queries = request.query.shape().dim(0);
  const uint64_t total_new = request.new_key.shape().dim(0);
  const uint64_t query_heads = request.query.shape().dim(1);
  const uint64_t kv_heads = request.new_key.shape().dim(1);
  const uint64_t head_dimension = request.query.shape().dim(2);
  const uint64_t batch = request.key_cache.shape().dim(0);
  const uint64_t max_context = request.key_cache.shape().dim(1);
  if (batch == 0 || query_heads == 0 || kv_heads == 0 || head_dimension == 0 ||
      query_heads % kv_heads != 0 || request.new_key.shape().dim(2) != head_dimension ||
      request.new_value.shape() != request.new_key.shape() ||
      request.key_cache.shape().dim(2) != kv_heads ||
      request.key_cache.shape().dim(3) != head_dimension ||
      request.value_cache.shape() != request.key_cache.shape() ||
      request.output.shape() != request.query.shape()) {
    return absl::InvalidArgumentError(
        "attention.shape: invalid Q/K/V, GQA head geometry, cache, or output shape");
  }
  if (request.query_indptr.size() != batch + 1 || request.new_kv_indptr.size() != batch + 1 ||
      request.kv_lengths_before.size() != batch ||
      request.query_positions.size() != total_queries) {
    return absl::InvalidArgumentError(
        "attention.metadata: metadata sizes do not match batch/tokens");
  }
  status = ValidateIndptr(request.query_indptr, total_queries,
                          "attention.query_indptr: invalid flattened offsets");
  if (!status.ok()) return status;
  status = ValidateIndptr(request.new_kv_indptr, total_new,
                          "attention.new_kv_indptr: invalid flattened offsets");
  if (!status.ok()) return status;
  for (size_t sequence = 0; sequence < static_cast<size_t>(batch); ++sequence) {
    const int32_t query_count = request.query_indptr[sequence + 1] - request.query_indptr[sequence];
    const int32_t new_count = request.new_kv_indptr[sequence + 1] - request.new_kv_indptr[sequence];
    const int32_t before = request.kv_lengths_before[sequence];
    if (before < 0 || query_count != new_count ||
        (request.phase == ExecutionPhase::kDecode && query_count > 1) ||
        static_cast<uint64_t>(before) + static_cast<uint64_t>(new_count) > max_context) {
      return absl::InvalidArgumentError(
          "attention.sequence: phase counts or KV capacity are invalid");
    }
    const size_t query_begin = static_cast<size_t>(request.query_indptr[sequence]);
    for (int32_t local = 0; local < query_count; ++local) {
      const int64_t expected = static_cast<int64_t>(before) + static_cast<int64_t>(local);
      if (static_cast<int64_t>(request.query_positions[query_begin + static_cast<size_t>(local)]) !=
          expected) {
        return absl::InvalidArgumentError(
            "attention.query_positions: expected contiguous absolute positions");
      }
    }
  }
  status = internal::ValidateSameDevice(request.query, request.new_key, "attention.device");
  if (!status.ok()) return status;
  status = internal::ValidateSameDevice(request.query, request.new_value, "attention.device");
  if (!status.ok()) return status;
  status = internal::ValidateSameDevice(request.query, request.key_cache, "attention.device");
  if (!status.ok()) return status;
  status = internal::ValidateSameDevice(request.query, request.value_cache, "attention.device");
  if (!status.ok()) return status;
  status = internal::ValidateSameDevice(request.query, request.output, "attention.device");
  if (!status.ok()) return status;

  status = RejectOverlap(request.query, request.new_key, "attention.alias: query/new key overlap");
  if (!status.ok()) return status;
  status =
      RejectOverlap(request.query, request.new_value, "attention.alias: query/new value overlap");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_key, request.new_value,
                         "attention.alias: new key/new value overlap");
  if (!status.ok()) return status;
  status =
      RejectOverlap(request.query, request.key_cache, "attention.alias: query/key cache overlap");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.value_cache,
                         "attention.alias: query/value cache overlap");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_key, request.key_cache,
                         "attention.alias: new key/key cache overlap");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_key, request.value_cache,
                         "attention.alias: new key/value cache overlap");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_value, request.key_cache,
                         "attention.alias: new value/key cache overlap");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_value, request.value_cache,
                         "attention.alias: new value/value cache overlap");
  if (!status.ok()) return status;
  status = RejectOverlap(request.key_cache, request.value_cache,
                         "attention.alias: key/value caches overlap");
  if (!status.ok()) return status;
  status = RejectOverlap(request.query, request.output, "attention.alias: query/output overlap");
  if (!status.ok()) return status;
  status = RejectOverlap(request.new_key, request.output, "attention.alias: key/output overlap");
  if (!status.ok()) return status;
  status =
      RejectOverlap(request.new_value, request.output, "attention.alias: value/output overlap");
  if (!status.ok()) return status;
  status =
      RejectOverlap(request.key_cache, request.output, "attention.alias: key cache/output overlap");
  if (!status.ok()) return status;
  status = RejectOverlap(request.value_cache, request.output,
                         "attention.alias: value cache/output overlap");
  if (!status.ok()) return status;
  return absl::OkStatus();
}

absl::Status ReferenceAttention(const AttentionRequest& request, std::span<float> scratch) {
  absl::Status status = ValidateAttention(request);
  if (!status.ok()) return status;
  const size_t max_context = static_cast<size_t>(request.key_cache.shape().dim(1));
  if (scratch.size() < max_context) {
    return absl::ResourceExhaustedError(
        "attention.scratch: one max-context FP32 score row is required");
  }
  absl::StatusOr<std::span<const float>> query = internal::HostSpan<float>(request.query);
  if (!query.ok()) return query.status();
  absl::StatusOr<std::span<const float>> new_key = internal::HostSpan<float>(request.new_key);
  if (!new_key.ok()) return new_key.status();
  absl::StatusOr<std::span<const float>> new_value = internal::HostSpan<float>(request.new_value);
  if (!new_value.ok()) return new_value.status();
  absl::StatusOr<std::span<float>> key_cache = internal::HostSpan<float>(request.key_cache);
  if (!key_cache.ok()) return key_cache.status();
  absl::StatusOr<std::span<float>> value_cache = internal::HostSpan<float>(request.value_cache);
  if (!value_cache.ok()) return value_cache.status();
  absl::StatusOr<std::span<float>> output = internal::HostSpan<float>(request.output);
  if (!output.ok()) return output.status();

  const size_t batch = static_cast<size_t>(request.key_cache.shape().dim(0));
  const size_t query_heads = static_cast<size_t>(request.query.shape().dim(1));
  const size_t kv_heads = static_cast<size_t>(request.new_key.shape().dim(1));
  const size_t head_dimension = static_cast<size_t>(request.query.shape().dim(2));
  const size_t group_size = query_heads / kv_heads;
  for (size_t sequence = 0; sequence < batch; ++sequence) {
    const size_t new_begin = static_cast<size_t>(request.new_kv_indptr[sequence]);
    const size_t new_end = static_cast<size_t>(request.new_kv_indptr[sequence + 1]);
    const size_t before = static_cast<size_t>(request.kv_lengths_before[sequence]);
    for (size_t local = 0; local < new_end - new_begin; ++local) {
      for (size_t head = 0; head < kv_heads; ++head) {
        for (size_t dimension = 0; dimension < head_dimension; ++dimension) {
          const size_t source =
              ((new_begin + local) * kv_heads + head) * head_dimension + dimension;
          const size_t destination =
              (((sequence * max_context + before + local) * kv_heads + head) * head_dimension) +
              dimension;
          (*key_cache)[destination] = (*new_key)[source];
          (*value_cache)[destination] = (*new_value)[source];
        }
      }
    }
  }

  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
  for (size_t sequence = 0; sequence < batch; ++sequence) {
    const size_t query_begin = static_cast<size_t>(request.query_indptr[sequence]);
    const size_t query_end = static_cast<size_t>(request.query_indptr[sequence + 1]);
    for (size_t query_index = query_begin; query_index < query_end; ++query_index) {
      const size_t position = static_cast<size_t>(request.query_positions[query_index]);
      const size_t visible = position + 1;
      for (size_t query_head = 0; query_head < query_heads; ++query_head) {
        const size_t kv_head = query_head / group_size;
        float maximum = -std::numeric_limits<float>::infinity();
        bool has_nan = false;
        for (size_t key_index = 0; key_index < visible; ++key_index) {
          float score = 0.0F;
          for (size_t dimension = 0; dimension < head_dimension; ++dimension) {
            const size_t query_offset =
                (query_index * query_heads + query_head) * head_dimension + dimension;
            const size_t key_offset =
                (((sequence * max_context + key_index) * kv_heads + kv_head) * head_dimension) +
                dimension;
            score += (*query)[query_offset] * (*key_cache)[key_offset];
          }
          scratch[key_index] = score * scale;
          has_nan = has_nan || std::isnan(scratch[key_index]);
          maximum = std::max(maximum, scratch[key_index]);
        }
        if (has_nan) maximum = std::numeric_limits<float>::quiet_NaN();
        float sum = 0.0F;
        for (size_t key_index = 0; key_index < visible; ++key_index) {
          scratch[key_index] = std::exp(scratch[key_index] - maximum);
          sum += scratch[key_index];
        }
        for (size_t dimension = 0; dimension < head_dimension; ++dimension) {
          float value = 0.0F;
          for (size_t key_index = 0; key_index < visible; ++key_index) {
            const size_t value_offset =
                (((sequence * max_context + key_index) * kv_heads + kv_head) * head_dimension) +
                dimension;
            value += (scratch[key_index] / sum) * (*value_cache)[value_offset];
          }
          const size_t output_offset =
              (query_index * query_heads + query_head) * head_dimension + dimension;
          (*output)[output_offset] = value;
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace inferx::ops
