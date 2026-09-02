#include "src/model_adapters/artifact_tensor_adapter.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/checked_math.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/dtype.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"

namespace inferx::model_adapters {
namespace {

class MappingDomain final : public AllocationDomain {
 public:
  MappingDomain(AllocationId id, artifacts::MappedTensorLease lease)
      : id_(id), lease_(std::move(lease)) {}

  absl::Status Release(void*, ByteCount, ByteCount, AllocationId id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_ || id != id_) {
      return absl::FailedPreconditionError("artifact_tensor_adapter: stale mapped buffer release");
    }
    lease_ = artifacts::MappedTensorLease();
    released_ = true;
    return absl::OkStatus();
  }

  void Abandon(void*, ByteCount, ByteCount, AllocationId id) noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_ || id != id_) return;
    lease_ = artifacts::MappedTensorLease();
    released_ = true;
  }

 private:
  AllocationId id_;
  artifacts::MappedTensorLease lease_;
  std::mutex mutex_;
  bool released_ = false;
};

absl::StatusOr<Dtype> ExecutionDtype(artifacts::ArtifactDtype dtype) {
  switch (dtype) {
    case artifacts::ArtifactDtype::kF16:
      return Dtype::kFloat16;
    case artifacts::ArtifactDtype::kBf16:
      return Dtype::kBFloat16;
    case artifacts::ArtifactDtype::kF32:
      return Dtype::kFloat32;
    default:
      return absl::UnimplementedError("artifact_tensor_adapter.dtype: artifact dtype unsupported");
  }
}

bool IsPowerOfTwo(uint64_t value) noexcept { return value != 0 && (value & (value - 1)) == 0; }

absl::StatusOr<AdaptedTensor> BuildAdapted(Buffer buffer, Dtype dtype,
                                           artifacts::ArtifactDtype source_dtype,
                                           const Shape& shape, bool staged) {
  absl::StatusOr<Strides> strides = Strides::Contiguous(shape);
  if (!strides.ok()) return strides.status();
  absl::StatusOr<BufferView> buffer_view = buffer.View(ByteRange{ByteCount(0), buffer.size()});
  if (!buffer_view.ok()) return buffer_view.status();
  absl::StatusOr<TensorView> view = TensorView::Create(*buffer_view, dtype, shape, *strides);
  if (!view.ok()) return view.status();
  return AdaptedTensor(std::move(buffer), *view, source_dtype, staged);
}

float HalfToFloat(uint16_t value) noexcept {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16U;
  uint32_t exponent = (value >> 10U) & 0x1fU;
  uint32_t fraction = value & 0x03ffU;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (fraction == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((fraction & 0x0400U) == 0) {
        fraction <<= 1U;
        ++shift;
      }
      fraction &= 0x03ffU;
      const uint32_t normalized_exponent = static_cast<uint32_t>(113 - shift);
      bits = sign | (normalized_exponent << 23U) | (fraction << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (fraction << 13U);
  } else {
    exponent += 112U;
    bits = sign | (exponent << 23U) | (fraction << 13U);
  }
  return std::bit_cast<float>(bits);
}

}  // namespace

absl::StatusOr<AdaptedTensor> ArtifactTensorAdapter::Adapt(const model::WeightPlanItem& plan,
                                                           const model::ParameterSpec& parameter,
                                                           artifacts::MappedTensorLease lease,
                                                           ByteCount required_alignment) const {
  if (plan.parameter != parameter.id || plan.source.shape != parameter.shape ||
      plan.transform.input_shape != parameter.shape ||
      plan.transform.output_shape != parameter.shape ||
      plan.transform.kind != model::TransformKind::kIdentity) {
    return absl::InvalidArgumentError(
        "artifact_tensor_adapter.parameter: plan identity/shape/transform mismatch");
  }
  if (std::find(parameter.allowed_source_dtypes.begin(), parameter.allowed_source_dtypes.end(),
                plan.source.dtype) == parameter.allowed_source_dtypes.end()) {
    return absl::InvalidArgumentError(
        "artifact_tensor_adapter.dtype: source dtype is not allowed by parameter contract");
  }
  if (!IsPowerOfTwo(required_alignment.value())) {
    return absl::InvalidArgumentError(
        "artifact_tensor_adapter.alignment: must be a nonzero power of two");
  }
  absl::StatusOr<Dtype> dtype = ExecutionDtype(plan.source.dtype);
  if (!dtype.ok()) return dtype.status();
  absl::StatusOr<Shape> shape = Shape::Create(parameter.shape);
  if (!shape.ok()) return shape.status();
  absl::StatusOr<ByteCount> exact_bytes = shape->Bytes(*dtype);
  if (!exact_bytes.ok()) return exact_bytes.status();
  absl::StatusOr<ByteCount> dtype_alignment = DtypeSize(*dtype);
  if (!dtype_alignment.ok()) return dtype_alignment.status();
  const ByteCount effective_alignment(
      std::max(required_alignment.value(), dtype_alignment->value()));
  if (lease.bytes().size() != exact_bytes->value() ||
      lease.requested_range() != plan.source.file_range ||
      !lease.file_identity().SameFileAndVersion(plan.source.expected_file)) {
    return absl::InvalidArgumentError(
        "artifact_tensor_adapter.bytes: mapped range differs from exact tensor bytes");
  }
  const uintptr_t address = reinterpret_cast<uintptr_t>(lease.bytes().data());
  if (exact_bytes->value() == 0 || address % effective_alignment.value() == 0) {
    const std::byte* mapped_data = lease.bytes().data();
    absl::StatusOr<AllocationId> id = NextAllocationId();
    if (!id.ok()) return id.status();
    std::shared_ptr<MappingDomain> domain;
    try {
      domain = std::make_shared<MappingDomain>(*id, std::move(lease));
    } catch (const std::bad_alloc&) {
      return absl::ResourceExhaustedError(
          "artifact_tensor_adapter: mapping owner allocation failed");
    }
    AllocationRequest request{Device::Host(), MemoryKind::kHost, *exact_bytes, effective_alignment,
                              MemoryCategory::kModelWeights};
    void* mutable_address =
        exact_bytes->value() == 0 ? nullptr : const_cast<std::byte*>(mapped_data);
    absl::StatusOr<Buffer> buffer =
        Buffer::Adopt(mutable_address, request, effective_alignment, *id, std::move(domain));
    if (!buffer.ok()) return buffer.status();
    return BuildAdapted(std::move(*buffer), *dtype, plan.source.dtype, *shape, false);
  }

  CpuAllocator allocator;
  absl::StatusOr<Buffer> staging =
      allocator.Allocate(AllocationRequest{Device::Host(), MemoryKind::kHost, *exact_bytes,
                                           effective_alignment, MemoryCategory::kModelWeights});
  if (!staging.ok()) return staging.status();
  absl::StatusOr<MutableBufferView> staging_view =
      staging->MutableView(ByteRange{ByteCount(0), *exact_bytes});
  if (!staging_view.ok()) return staging_view.status();
  absl::StatusOr<std::span<std::byte>> destination = staging_view->HostBytes();
  if (!destination.ok()) return destination.status();
  std::memcpy(destination->data(), lease.bytes().data(), lease.bytes().size());
  return BuildAdapted(std::move(*staging), *dtype, plan.source.dtype, *shape, true);
}

absl::StatusOr<AdaptedTensor> ReferenceTensorMaterializer::Materialize(
    const AdaptedTensor& input, uint64_t maximum_elements) const {
  absl::StatusOr<uint64_t> elements = input.view().shape().NumElements();
  if (!elements.ok()) return elements.status();
  if (*elements > maximum_elements) {
    return absl::ResourceExhaustedError("reference_materializer: element limit exceeded");
  }
  absl::StatusOr<ByteCount> bytes = input.view().shape().Bytes(Dtype::kFloat32);
  if (!bytes.ok()) return bytes.status();
  CpuAllocator allocator;
  absl::StatusOr<Buffer> output = allocator.Allocate(AllocationRequest{
      Device::Host(), MemoryKind::kHost, *bytes, ByteCount(alignof(float)), MemoryCategory::kTest});
  if (!output.ok()) return output.status();
  absl::StatusOr<MutableBufferView> output_view =
      output->MutableView(ByteRange{ByteCount(0), *bytes});
  if (!output_view.ok()) return output_view.status();
  if (*elements == 0) {
    return BuildAdapted(std::move(*output), Dtype::kFloat32, input.source_dtype(),
                        input.view().shape(), true);
  }
  absl::StatusOr<std::span<std::byte>> output_bytes = output_view->HostBytes();
  if (!output_bytes.ok()) return output_bytes.status();
  auto* floats = reinterpret_cast<float*>(output_bytes->data());
  absl::StatusOr<std::span<const std::byte>> source = input.view().buffer().HostBytes();
  if (!source.ok()) return source.status();
  const std::byte* source_data = source->data() + input.view().byte_offset().value();
  if (input.view().dtype() == Dtype::kFloat32) {
    std::memcpy(floats, source_data, static_cast<size_t>(bytes->value()));
  } else if (input.view().dtype() == Dtype::kFloat16 || input.view().dtype() == Dtype::kBFloat16) {
    for (size_t index = 0; index < static_cast<size_t>(*elements); ++index) {
      uint16_t word = 0;
      std::memcpy(&word, source_data + index * sizeof(word), sizeof(word));
      floats[index] = input.view().dtype() == Dtype::kFloat16
                          ? HalfToFloat(word)
                          : std::bit_cast<float>(static_cast<uint32_t>(word) << 16U);
    }
  } else {
    return absl::UnimplementedError("reference_materializer: unsupported execution dtype");
  }
  return BuildAdapted(std::move(*output), Dtype::kFloat32, input.source_dtype(),
                      input.view().shape(), true);
}

absl::StatusOr<std::vector<inferx::ops::KernelKey>> BuildRequiredKernelSet(
    const model::ModelSpec& model, const inferx::ops::OperatorEnvelope& envelope) {
  const model::LlamaSpec& llama = model.llama();
  return inferx::ops::BuildRequiredKernelSet(
      inferx::ops::LlamaOperatorSpec{llama.vocab_size, llama.hidden_size, llama.intermediate_size,
                                     llama.num_attention_heads, llama.num_key_value_heads,
                                     llama.head_dim},
      envelope);
}

}  // namespace inferx::model_adapters
