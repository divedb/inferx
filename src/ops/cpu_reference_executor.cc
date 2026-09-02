#include "inferx/ops/cpu_reference_executor.h"

#include <cstdint>
#include <optional>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/checked_math.h"

namespace inferx::ops {
namespace {

absl::StatusOr<uint64_t> Elements(const Shape& shape) { return shape.NumElements(); }

absl::StatusOr<uint64_t> GemmFlops(const GemmRequest& request) {
  absl::StatusOr<uint64_t> mn = CheckedMul(request.input.shape().dim(0),
                                           request.weight.shape().dim(0), "reference.gemm.flops");
  if (!mn.ok()) return mn.status();
  absl::StatusOr<uint64_t> mnk =
      CheckedMul(*mn, request.input.shape().dim(1), "reference.gemm.flops");
  if (!mnk.ok()) return mnk.status();
  return CheckedMul(*mnk, uint64_t{2}, "reference.gemm.flops");
}

}  // namespace

absl::Status CpuReferenceExecutor::CheckElements(uint64_t elements) const {
  return elements <= limits_.maximum_elements
             ? absl::OkStatus()
             : absl::ResourceExhaustedError("reference.elements: configured limit exceeded");
}

absl::Status CpuReferenceExecutor::CheckFlops(uint64_t flops) const {
  return flops <= limits_.maximum_flops
             ? absl::OkStatus()
             : absl::ResourceExhaustedError("reference.flops: configured limit exceeded");
}

absl::Status CpuReferenceExecutor::Execute(const EmbeddingRequest& request) const {
  absl::Status status = ValidateEmbedding(request);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> elements = Elements(request.output.shape());
  if (!elements.ok()) return elements.status();
  status = CheckElements(*elements);
  return status.ok() ? ReferenceEmbedding(request) : status;
}

absl::Status CpuReferenceExecutor::Execute(const GemmRequest& request) const {
  absl::Status status = ValidateGemm(request);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> elements = Elements(request.output.shape());
  if (!elements.ok()) return elements.status();
  status = CheckElements(*elements);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> flops = GemmFlops(request);
  if (!flops.ok()) return flops.status();
  status = CheckFlops(*flops);
  return status.ok() ? ReferenceGemm(request) : status;
}

absl::Status CpuReferenceExecutor::Execute(const RmsNormRequest& request) const {
  absl::Status status = ValidateRmsNorm(request);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> elements = Elements(request.output.shape());
  if (!elements.ok()) return elements.status();
  status = CheckElements(*elements);
  return status.ok() ? ReferenceRmsNorm(request) : status;
}

absl::Status CpuReferenceExecutor::Execute(const RopeRequest& request) const {
  absl::Status status = ValidateRope(request);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> query_elements = Elements(request.query_output.shape());
  if (!query_elements.ok()) return query_elements.status();
  absl::StatusOr<uint64_t> key_elements = Elements(request.key_output.shape());
  if (!key_elements.ok()) return key_elements.status();
  absl::StatusOr<uint64_t> total =
      CheckedAdd(*query_elements, *key_elements, "reference.rope.elements");
  if (!total.ok()) return total.status();
  status = CheckElements(*total);
  return status.ok() ? ReferenceRope(request) : status;
}

absl::Status CpuReferenceExecutor::Execute(const SiluRequest& request) const {
  absl::Status status = ValidateSilu(request);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> elements = Elements(request.output.shape());
  if (!elements.ok()) return elements.status();
  status = CheckElements(*elements);
  return status.ok() ? ReferenceSilu(request) : status;
}

absl::Status CpuReferenceExecutor::ExecuteMultiply(const MultiplyRequest& request) const {
  absl::Status status = ValidateMultiply(request);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> elements = Elements(request.output.shape());
  if (!elements.ok()) return elements.status();
  status = CheckElements(*elements);
  return status.ok() ? ReferenceMultiply(request) : status;
}

absl::Status CpuReferenceExecutor::ExecuteSwiGlu(const SwiGluRequest& request) const {
  absl::Status status = ValidateSwiGlu(request);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> elements = Elements(request.output.shape());
  if (!elements.ok()) return elements.status();
  status = CheckElements(*elements);
  return status.ok() ? ReferenceSwiGlu(request) : status;
}

absl::Status CpuReferenceExecutor::ExecuteResidual(const ResidualRequest& request) const {
  absl::Status status = ValidateResidual(request);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> elements = Elements(request.output.shape());
  if (!elements.ok()) return elements.status();
  status = CheckElements(*elements);
  return status.ok() ? ReferenceResidual(request) : status;
}

absl::Status CpuReferenceExecutor::Execute(const AttentionRequest& request,
                                           std::span<float> scratch) const {
  absl::Status status = ValidateAttention(request);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> elements = Elements(request.output.shape());
  if (!elements.ok()) return elements.status();
  status = CheckElements(*elements);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> score_work =
      CheckedMul(*elements, request.key_cache.shape().dim(1), "reference.attention.flops");
  if (!score_work.ok()) return score_work.status();
  absl::StatusOr<uint64_t> flops =
      CheckedMul(*score_work, uint64_t{4}, "reference.attention.flops");
  if (!flops.ok()) return flops.status();
  status = CheckFlops(*flops);
  return status.ok() ? ReferenceAttention(request, scratch) : status;
}

absl::Status CpuReferenceExecutor::Execute(const LogitsRequest& request) const {
  absl::Status status = ValidateLogits(request);
  if (!status.ok()) return status;
  GemmRequest gemm{request.hidden, request.weight, std::nullopt, request.output, 1.0F, 0.0F};
  absl::StatusOr<uint64_t> elements = Elements(request.output.shape());
  if (!elements.ok()) return elements.status();
  status = CheckElements(*elements);
  if (!status.ok()) return status;
  absl::StatusOr<uint64_t> flops = GemmFlops(gemm);
  if (!flops.ok()) return flops.status();
  status = CheckFlops(*flops);
  return status.ok() ? ReferenceLogits(request) : status;
}

}  // namespace inferx::ops
