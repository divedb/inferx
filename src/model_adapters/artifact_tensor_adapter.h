#ifndef INFERX_SRC_MODEL_ADAPTERS_ARTIFACT_TENSOR_ADAPTER_H_
#define INFERX_SRC_MODEL_ADAPTERS_ARTIFACT_TENSOR_ADAPTER_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "inferx/artifacts/mapped_region.h"
#include "inferx/model/model_spec.h"
#include "inferx/model/parameter_spec.h"
#include "inferx/model/weight_plan.h"
#include "inferx/ops/warmup.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::model_adapters {

class AdaptedTensor {
 public:
  AdaptedTensor(AdaptedTensor&&) noexcept = default;
  AdaptedTensor& operator=(AdaptedTensor&&) noexcept = default;
  AdaptedTensor(const AdaptedTensor&) = delete;
  AdaptedTensor& operator=(const AdaptedTensor&) = delete;

  [[nodiscard]] const TensorView& view() const noexcept { return view_; }
  [[nodiscard]] artifacts::ArtifactDType source_dtype() const noexcept { return source_dtype_; }
  [[nodiscard]] bool staged() const noexcept { return staged_; }
  [[nodiscard]] absl::Status Release() { return buffer_.Release(); }

  AdaptedTensor(Buffer buffer, TensorView view, artifacts::ArtifactDType source_dtype,
                bool staged) noexcept
      : buffer_(std::move(buffer)),
        view_(std::move(view)),
        source_dtype_(source_dtype),
        staged_(staged) {}

 private:
  Buffer buffer_;
  TensorView view_;
  artifacts::ArtifactDType source_dtype_ = artifacts::ArtifactDType::kF32;
  bool staged_ = false;
};

class ArtifactTensorAdapter {
 public:
  [[nodiscard]] absl::StatusOr<AdaptedTensor> Adapt(const model::WeightPlanItem& plan,
                                                    const model::ParameterSpec& parameter,
                                                    artifacts::MappedTensorLease lease,
                                                    ByteCount required_alignment) const;
};

class ReferenceTensorMaterializer {
 public:
  [[nodiscard]] absl::StatusOr<AdaptedTensor> Materialize(const AdaptedTensor& input,
                                                          uint64_t maximum_elements) const;
};

[[nodiscard]] absl::StatusOr<std::vector<inferx::ops::KernelKey>> BuildRequiredKernelSet(
    const model::ModelSpec& model, const inferx::ops::OperatorEnvelope& envelope);

}  // namespace inferx::model_adapters

#endif  // INFERX_SRC_MODEL_ADAPTERS_ARTIFACT_TENSOR_ADAPTER_H_
