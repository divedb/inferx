#include "inferx/platform/cuda/ops/cublaslt_gemm.h"

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"
#include "inferx/platform/cuda/cuda_buffer_access.h"
#include "inferx/platform/cuda/cuda_error.h"

namespace inferx::cuda::ops {
namespace {

absl::Status CublasStatus(cublasStatus_t status, std::string_view operation) {
  if (status == CUBLAS_STATUS_SUCCESS) return absl::OkStatus();
  absl::StatusCode code = absl::StatusCode::kInternal;
  if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) {
    code = absl::StatusCode::kUnimplemented;
  } else if (status == CUBLAS_STATUS_ALLOC_FAILED) {
    code = absl::StatusCode::kResourceExhausted;
  } else if (status == CUBLAS_STATUS_INVALID_VALUE) {
    code = absl::StatusCode::kInvalidArgument;
  }
  return absl::Status(code,
                      absl::StrCat("cublaslt.", operation, ": status ", static_cast<int>(status)));
}

absl::StatusOr<cudaDataType_t> CudaDataType(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return CUDA_R_32F;
    case DType::kFloat16:
      return CUDA_R_16F;
    case DType::kBFloat16:
      return CUDA_R_16BF;
    default:
      return absl::UnimplementedError("cublaslt.dtype: FP32, FP16, or BF16 required");
  }
}

const void* Address(const TensorView& tensor) noexcept {
  const auto* base = static_cast<const std::byte*>(BufferAccess::Address(tensor.buffer()));
  return base == nullptr ? nullptr : base + tensor.byte_offset().value();
}

void* Address(const MutableTensorView& tensor) noexcept {
  auto* base = static_cast<std::byte*>(BufferAccess::Address(tensor.buffer()));
  return base == nullptr ? nullptr : base + tensor.byte_offset().value();
}

void* Address(const MutableBufferView& buffer) noexcept { return BufferAccess::Address(buffer); }

struct TensorInterval {
  AllocationId allocation;
  uint64_t begin = 0;
  uint64_t size = 0;
};

template <typename View>
absl::StatusOr<TensorInterval> Interval(const View& tensor) {
  absl::StatusOr<uint64_t> begin =
      CheckedAdd(tensor.buffer().range().offset.value(), tensor.byte_offset().value(),
                 "cublaslt.tensor_interval");
  if (!begin.ok()) return begin.status();
  return TensorInterval{tensor.buffer().allocation_id(), *begin,
                        tensor.layout().reachable_bytes.value()};
}

bool Overlaps(const TensorInterval& left, const TensorInterval& right) noexcept {
  if (left.allocation != right.allocation || left.size == 0 || right.size == 0) return false;
  if (left.begin <= right.begin) return right.begin - left.begin < left.size;
  return left.begin - right.begin < right.size;
}

template <typename Left, typename Right>
absl::Status RequireDisjoint(const Left& left, const Right& right, std::string_view field) {
  absl::StatusOr<TensorInterval> left_interval = Interval(left);
  if (!left_interval.ok()) return left_interval.status();
  absl::StatusOr<TensorInterval> right_interval = Interval(right);
  if (!right_interval.ok()) return right_interval.status();
  return Overlaps(*left_interval, *right_interval)
             ? absl::InvalidArgumentError(absl::StrCat(field, ": operands overlap"))
             : absl::OkStatus();
}

TensorInterval Interval(const MutableBufferView& buffer) noexcept {
  return TensorInterval{buffer.allocation_id(), buffer.range().offset.value(),
                        buffer.range().size.value()};
}

template <typename View>
absl::Status RequireDisjointWorkspace(const View& tensor, const MutableBufferView& workspace,
                                      std::string_view field) {
  absl::StatusOr<TensorInterval> tensor_interval = Interval(tensor);
  if (!tensor_interval.ok()) return tensor_interval.status();
  return Overlaps(*tensor_interval, Interval(workspace))
             ? absl::InvalidArgumentError(absl::StrCat(field, ": workspace overlaps operand"))
             : absl::OkStatus();
}

template <typename View>
bool IsDeviceOperand(const View& tensor, DeviceId device) noexcept {
  return tensor.buffer().device() == Device::Cuda(device) &&
         tensor.buffer().memory_kind() == MemoryKind::kDevice;
}

template <typename View>
bool IsAligned(const View& tensor, uint32_t alignment) noexcept {
  if (tensor.layout().reachable_bytes.value() == 0) return true;
  return reinterpret_cast<uintptr_t>(Address(tensor)) % alignment == 0;
}

absl::Status SetOrder(cublasLtMatrixLayout_t layout) {
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  return CublasStatus(
      cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
      "matrix_layout_set_order");
}

absl::Status SetSingleBatch(cublasLtMatrixLayout_t layout) {
  const int32_t batch_count = 1;
  absl::Status status =
      CublasStatus(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                    &batch_count, sizeof(batch_count)),
                   "matrix_layout_set_batch_count");
  if (!status.ok()) return status;
  const int64_t batch_stride = 0;
  return CublasStatus(
      cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                       &batch_stride, sizeof(batch_stride)),
      "matrix_layout_set_batch_stride");
}

absl::Status ConfigureLayout(cublasLtMatrixLayout_t layout) {
  absl::Status status = SetOrder(layout);
  return status.ok() ? SetSingleBatch(layout) : status;
}

}  // namespace

struct CublasLtHandleState {
  cublasLtHandle_t handle = nullptr;
  DeviceId device = DeviceId(0);
  uint16_t compute_capability = 0;
  CudaHealth* health = nullptr;

  ~CublasLtHandleState() noexcept {
    if (handle != nullptr) static_cast<void>(cublasLtDestroy(handle));
  }
};

struct CublasLtGemmPlan::Impl {
  std::shared_ptr<CublasLtHandleState> owner;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t input = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t addend = nullptr;
  cublasLtMatrixLayout_t output = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  inferx::ops::KernelKey key;
  ByteCount workspace = ByteCount(0);

  ~Impl() noexcept {
    if (output != nullptr) static_cast<void>(cublasLtMatrixLayoutDestroy(output));
    if (addend != nullptr) static_cast<void>(cublasLtMatrixLayoutDestroy(addend));
    if (weight != nullptr) static_cast<void>(cublasLtMatrixLayoutDestroy(weight));
    if (input != nullptr) static_cast<void>(cublasLtMatrixLayoutDestroy(input));
    if (operation != nullptr) static_cast<void>(cublasLtMatmulDescDestroy(operation));
  }
};

struct CublasLtContext::Impl {
  std::shared_ptr<CublasLtHandleState> state;
};

CublasLtGemmPlan::CublasLtGemmPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
CublasLtGemmPlan::~CublasLtGemmPlan() noexcept = default;
CublasLtGemmPlan::CublasLtGemmPlan(CublasLtGemmPlan&&) noexcept = default;
CublasLtGemmPlan& CublasLtGemmPlan::operator=(CublasLtGemmPlan&&) noexcept = default;

ByteCount CublasLtGemmPlan::workspace_bytes() const noexcept {
  return impl_ == nullptr ? ByteCount(0) : impl_->workspace;
}

absl::Status CublasLtGemmPlan::Launch(const inferx::ops::GemmRequest& request,
                                      const CudaOpContext& context,
                                      MutableBufferView workspace) const {
  if (impl_ == nullptr) return absl::FailedPreconditionError("cublaslt.plan: moved plan");
  if (impl_->owner == nullptr || impl_->owner->handle == nullptr) {
    return absl::FailedPreconditionError("cublaslt.plan: owning context is closed");
  }
  if (context.device != impl_->owner->device || context.stream.device() != impl_->owner->device) {
    return absl::InvalidArgumentError("cublaslt.launch: context device mismatch");
  }
  absl::Status status = impl_->owner->health == nullptr
                            ? absl::OkStatus()
                            : impl_->owner->health->CheckAcceptingWork();
  if (!status.ok()) return status;
  if (request.input.shape().rank() != 2 || request.weight.shape().rank() != 2 ||
      request.output.shape().rank() != 2 || !request.input.layout().contiguous ||
      !request.weight.layout().contiguous || !request.output.layout().contiguous ||
      request.input.shape().dim(0) != impl_->key.dimensions[0] ||
      request.input.shape().dim(1) != impl_->key.dimensions[1] ||
      request.weight.shape().dim(0) != impl_->key.dimensions[2] ||
      request.weight.shape().dim(1) != impl_->key.dimensions[1] ||
      request.output.shape().dim(0) != impl_->key.dimensions[0] ||
      request.output.shape().dim(1) != impl_->key.dimensions[2] ||
      request.input.dtype() != impl_->key.input_dtype ||
      request.weight.dtype() != impl_->key.weight_dtype ||
      request.output.dtype() != impl_->key.output_dtype) {
    return absl::InvalidArgumentError("cublaslt.launch: request differs from prepared key");
  }
  if (!std::isfinite(request.alpha) || !std::isfinite(request.beta) ||
      (request.beta != 0.0F && !request.addend.has_value())) {
    return absl::InvalidArgumentError("cublaslt.launch: invalid alpha/beta/addend contract");
  }
  if (!IsDeviceOperand(request.input, impl_->owner->device) ||
      !IsDeviceOperand(request.weight, impl_->owner->device) ||
      !IsDeviceOperand(request.output, impl_->owner->device)) {
    return absl::InvalidArgumentError("cublaslt.launch: CUDA operand device mismatch");
  }
  if (request.addend.has_value() && (request.addend->shape() != request.output.shape() ||
                                     request.addend->dtype() != request.output.dtype() ||
                                     !IsDeviceOperand(*request.addend, impl_->owner->device))) {
    return absl::InvalidArgumentError("cublaslt.launch: invalid addend");
  }
  status = RequireDisjoint(request.input, request.output, "cublaslt.alias.input_output");
  if (!status.ok()) return status;
  status = RequireDisjoint(request.weight, request.output, "cublaslt.alias.weight_output");
  if (!status.ok()) return status;
  status = RequireDisjoint(request.input, request.weight, "cublaslt.alias.input_weight");
  if (!status.ok()) return status;
  if (request.addend.has_value()) {
    status = RequireDisjoint(*request.addend, request.output, "cublaslt.alias.addend_output");
    if (!status.ok()) return status;
    status = RequireDisjoint(request.input, *request.addend, "cublaslt.alias.input_addend");
    if (!status.ok()) return status;
    status = RequireDisjoint(request.weight, *request.addend, "cublaslt.alias.weight_addend");
    if (!status.ok()) return status;
  }
  if (!IsAligned(request.input, impl_->key.alignment_class) ||
      !IsAligned(request.weight, impl_->key.alignment_class) ||
      !IsAligned(request.output, impl_->key.alignment_class) ||
      (request.addend.has_value() && !IsAligned(*request.addend, impl_->key.alignment_class))) {
    return absl::InvalidArgumentError("cublaslt.launch: operand alignment differs from key");
  }
  if (workspace.size().value() < impl_->workspace.value() ||
      (impl_->workspace.value() != 0 && (workspace.alignment().value() < 256 ||
                                         workspace.device() != Device::Cuda(impl_->owner->device) ||
                                         workspace.memory_kind() != MemoryKind::kDevice))) {
    return absl::ResourceExhaustedError("cublaslt.launch: workspace is insufficient or unaligned");
  }
  if (impl_->workspace.value() != 0) {
    status = RequireDisjointWorkspace(request.input, workspace, "cublaslt.workspace.input");
    if (!status.ok()) return status;
    status = RequireDisjointWorkspace(request.weight, workspace, "cublaslt.workspace.weight");
    if (!status.ok()) return status;
    status = RequireDisjointWorkspace(request.output, workspace, "cublaslt.workspace.output");
    if (!status.ok()) return status;
    if (request.addend.has_value()) {
      status = RequireDisjointWorkspace(*request.addend, workspace, "cublaslt.workspace.addend");
      if (!status.ok()) return status;
    }
  }
  if (impl_->key.dimensions[0] == 0) return absl::OkStatus();
  const void* addend =
      request.addend.has_value() ? Address(*request.addend) : Address(request.output);
  const cublasStatus_t launch_status =
      cublasLtMatmul(impl_->owner->handle, impl_->operation, &request.alpha, Address(request.input),
                     impl_->input, Address(request.weight), impl_->weight, &request.beta, addend,
                     impl_->addend, Address(request.output), impl_->output, &impl_->algorithm,
                     impl_->workspace.value() == 0 ? nullptr : Address(workspace),
                     static_cast<size_t>(impl_->workspace.value()), context.stream.handle());
  status = CublasStatus(launch_status, "matmul");
  if (!status.ok() && context.health != nullptr &&
      (launch_status == CUBLAS_STATUS_EXECUTION_FAILED ||
       launch_status == CUBLAS_STATUS_INTERNAL_ERROR)) {
    context.health->Poison(status);
  }
  if (!status.ok()) return status;
  return CheckCuda(cudaPeekAtLastError(), "ops.cublaslt.peek", context.device, context.health);
}

absl::Status CublasLtGemmPlan::Launch(const inferx::ops::LogitsRequest& request,
                                      const CudaOpContext& context,
                                      MutableBufferView workspace) const {
  return Launch(inferx::ops::GemmRequest{request.hidden, request.weight, std::nullopt,
                                         request.output, 1.0F, 0.0F},
                context, workspace);
}

CublasLtContext::CublasLtContext(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
CublasLtContext::~CublasLtContext() noexcept = default;
CublasLtContext::CublasLtContext(CublasLtContext&&) noexcept = default;
CublasLtContext& CublasLtContext::operator=(CublasLtContext&&) noexcept = default;

absl::StatusOr<CublasLtContext> CublasLtContext::Create(DeviceId device, CudaHealth* health) {
  std::unique_ptr<Impl> impl;
  try {
    impl = std::make_unique<Impl>();
    impl->state = std::make_shared<CublasLtHandleState>();
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("cublaslt.context: host allocation failed");
  }
  impl->state->device = device;
  impl->state->health = health;
  absl::Status status = CheckCuda(cudaSetDevice(static_cast<int>(device.value())),
                                  "ops.cublaslt.set_device", device, health);
  if (!status.ok()) return status;
  cudaDeviceProp properties{};
  status = CheckCuda(cudaGetDeviceProperties(&properties, static_cast<int>(device.value())),
                     "ops.cublaslt.get_device_properties", device, health);
  if (!status.ok()) return status;
  if (properties.major <= 0 || properties.minor < 0 || properties.minor > 9) {
    return absl::FailedPreconditionError("cublaslt.context: invalid device compute capability");
  }
  const int compute_capability = properties.major * 10 + properties.minor;
  if (compute_capability > static_cast<int>(std::numeric_limits<uint16_t>::max())) {
    return absl::FailedPreconditionError(
        "cublaslt.context: device compute capability exceeds key representation");
  }
  impl->state->compute_capability = static_cast<uint16_t>(compute_capability);
  status = CublasStatus(cublasLtCreate(&impl->state->handle), "create");
  if (!status.ok()) return status;
  return CublasLtContext(std::move(impl));
}

absl::StatusOr<CublasLtGemmPlan> CublasLtContext::Prepare(const inferx::ops::KernelKey& key) const {
  if (impl_ == nullptr || impl_->state == nullptr || impl_->state->handle == nullptr) {
    return absl::FailedPreconditionError("cublaslt.context: context is closed or moved");
  }
  absl::Status status = inferx::ops::ValidateKernelKey(key);
  if (!status.ok()) return status;
  if (key.op != inferx::ops::OpKind::kGemm && key.op != inferx::ops::OpKind::kLogits) {
    return absl::InvalidArgumentError("cublaslt.prepare: GEMM or logits key required");
  }
  if (key.device_kind != DeviceKind::kCuda || key.rank != 3 || key.dimensions[1] == 0 ||
      key.dimensions[2] == 0 || key.input_layout != inferx::ops::LayoutId::kRowMajorDense ||
      key.weight_layout != inferx::ops::LayoutId::kRowMajorDense ||
      key.output_layout != inferx::ops::LayoutId::kRowMajorDense ||
      key.input_dtype != key.weight_dtype) {
    return absl::UnimplementedError("cublaslt.prepare: unsupported key layout or dtype family");
  }
  if (key.compute_capability != impl_->state->compute_capability) {
    return absl::UnimplementedError("cublaslt.prepare: key SM differs from owning device");
  }
  if (key.op == inferx::ops::OpKind::kLogits && key.output_dtype != DType::kFloat32) {
    return absl::UnimplementedError("cublaslt.prepare: logits output must use FP32 storage");
  }
  absl::StatusOr<cudaDataType_t> input_type = CudaDataType(key.input_dtype);
  if (!input_type.ok()) return input_type.status();
  absl::StatusOr<cudaDataType_t> output_type = CudaDataType(key.output_dtype);
  if (!output_type.ok()) return output_type.status();
  absl::StatusOr<ByteCount> input_size = DTypeSize(key.input_dtype);
  if (!input_size.ok()) return input_size.status();
  absl::StatusOr<ByteCount> output_size = DTypeSize(key.output_dtype);
  if (!output_size.ok()) return output_size.status();
  if (key.alignment_class < std::max(input_size->value(), output_size->value()) ||
      key.dimensions[1] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      key.dimensions[2] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      key.workspace_limit_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return absl::UnimplementedError("cublaslt.prepare: alignment or vendor integer limit");
  }
  std::unique_ptr<CublasLtGemmPlan::Impl> plan;
  try {
    plan = std::make_unique<CublasLtGemmPlan::Impl>();
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("cublaslt.prepare: host allocation failed");
  }
  plan->owner = impl_->state;
  plan->key = key;
  const uint64_t rows = key.dimensions[0];
  const uint64_t inner = key.dimensions[1];
  const uint64_t columns = key.dimensions[2];
  if (rows == 0) return CublasLtGemmPlan(std::move(plan));
  const cublasComputeType_t compute =
      key.input_dtype == DType::kFloat32 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
  status = CublasStatus(cublasLtMatmulDescCreate(&plan->operation, compute, CUDA_R_32F),
                        "matmul_desc_create");
  if (!status.ok()) return status;
  const cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  status = CublasStatus(
      cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                     &pointer_mode, sizeof(pointer_mode)),
      "matmul_desc_set_pointer_mode");
  if (!status.ok()) return status;
  const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  status =
      CublasStatus(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &epilogue, sizeof(epilogue)),
                   "matmul_desc_set_epilogue");
  if (!status.ok()) return status;
  const cublasOperation_t transpose_input = CUBLAS_OP_N;
  const cublasOperation_t transpose_weight = CUBLAS_OP_T;
  status = CublasStatus(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                       &transpose_input, sizeof(transpose_input)),
                        "matmul_desc_set_transa");
  if (!status.ok()) return status;
  status = CublasStatus(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                       &transpose_weight, sizeof(transpose_weight)),
                        "matmul_desc_set_transb");
  if (!status.ok()) return status;
  status = CublasStatus(cublasLtMatrixLayoutCreate(&plan->input, *input_type, rows, inner,
                                                   static_cast<int64_t>(inner)),
                        "input_layout_create");
  if (!status.ok()) return status;
  status = ConfigureLayout(plan->input);
  if (!status.ok()) return status;
  status = CublasStatus(cublasLtMatrixLayoutCreate(&plan->weight, *input_type, columns, inner,
                                                   static_cast<int64_t>(inner)),
                        "weight_layout_create");
  if (!status.ok()) return status;
  status = ConfigureLayout(plan->weight);
  if (!status.ok()) return status;
  status = CublasStatus(cublasLtMatrixLayoutCreate(&plan->addend, *output_type, rows, columns,
                                                   static_cast<int64_t>(columns)),
                        "addend_layout_create");
  if (!status.ok()) return status;
  status = ConfigureLayout(plan->addend);
  if (!status.ok()) return status;
  status = CublasStatus(cublasLtMatrixLayoutCreate(&plan->output, *output_type, rows, columns,
                                                   static_cast<int64_t>(columns)),
                        "output_layout_create");
  if (!status.ok()) return status;
  status = ConfigureLayout(plan->output);
  if (!status.ok()) return status;

  cublasLtMatmulPreference_t preference = nullptr;
  status = CublasStatus(cublasLtMatmulPreferenceCreate(&preference), "preference_create");
  if (!status.ok()) return status;
  const size_t workspace_limit = static_cast<size_t>(key.workspace_limit_bytes);
  status = CublasStatus(
      cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &workspace_limit, sizeof(workspace_limit)),
      "preference_set_workspace");
  if (!status.ok()) {
    static_cast<void>(cublasLtMatmulPreferenceDestroy(preference));
    return status;
  }
  const uint32_t minimum_alignment = key.alignment_class;
  for (const cublasLtMatmulPreferenceAttributes_t attribute :
       {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
    status = CublasStatus(cublasLtMatmulPreferenceSetAttribute(
                              preference, attribute, &minimum_alignment, sizeof(minimum_alignment)),
                          "preference_set_operand_alignment");
    if (!status.ok()) {
      static_cast<void>(cublasLtMatmulPreferenceDestroy(preference));
      return status;
    }
  }
  std::array<cublasLtMatmulHeuristicResult_t, 8> candidates{};
  int candidate_count = 0;
  status =
      CublasStatus(cublasLtMatmulAlgoGetHeuristic(
                       impl_->state->handle, plan->operation, plan->input, plan->weight,
                       plan->addend, plan->output, preference, static_cast<int>(candidates.size()),
                       candidates.data(), &candidate_count),
                   "algo_get_heuristic");
  const absl::Status destroy_preference =
      CublasStatus(cublasLtMatmulPreferenceDestroy(preference), "preference_destroy");
  if (!status.ok()) return status;
  if (!destroy_preference.ok()) return destroy_preference;
  bool found = false;
  for (int index = 0; index < candidate_count; ++index) {
    if (candidates[static_cast<size_t>(index)].state != CUBLAS_STATUS_SUCCESS ||
        candidates[static_cast<size_t>(index)].workspaceSize > key.workspace_limit_bytes) {
      continue;
    }
    cublasLtMatmulHeuristicResult_t checked{};
    status = CublasStatus(
        cublasLtMatmulAlgoCheck(impl_->state->handle, plan->operation, plan->input, plan->weight,
                                plan->addend, plan->output,
                                &candidates[static_cast<size_t>(index)].algo, &checked),
        "algo_check");
    if (!status.ok() || checked.state != CUBLAS_STATUS_SUCCESS ||
        checked.workspaceSize > key.workspace_limit_bytes) {
      continue;
    }
    plan->algorithm = checked.algo;
    plan->workspace = ByteCount(static_cast<uint64_t>(checked.workspaceSize));
    found = true;
    break;
  }
  if (!found) return absl::UnimplementedError("cublaslt.prepare: no valid bounded algorithm");
  return CublasLtGemmPlan(std::move(plan));
}

absl::Status CublasLtContext::Close() {
  if (impl_ == nullptr || impl_->state == nullptr || impl_->state->handle == nullptr) {
    return absl::OkStatus();
  }
  if (impl_->state.use_count() != 1) {
    return absl::FailedPreconditionError(
        "cublaslt.context: prepared plans must be destroyed before close");
  }
  absl::Status status = CublasStatus(cublasLtDestroy(impl_->state->handle), "destroy");
  if (status.ok()) impl_->state->handle = nullptr;
  return status;
}

}  // namespace inferx::cuda::ops
