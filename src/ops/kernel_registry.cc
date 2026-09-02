#include "inferx/ops/kernel_registry.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"

namespace inferx::ops {

absl::Status BackendCapability::Validate() const {
  if (capability_id.empty()) {
    return absl::InvalidArgumentError("kernel_capability.id: must not be empty");
  }
  if (phase_mask == 0 || input_dtype_mask == 0 || weight_dtype_mask == 0 ||
      output_dtype_mask == 0 || alias_mask == 0) {
    return absl::InvalidArgumentError("kernel_capability.mask: supported sets must not be empty");
  }
  constexpr uint8_t kKnownPhases = PhaseMask(ExecutionPhase::kGeneric) |
                                   PhaseMask(ExecutionPhase::kPrefill) |
                                   PhaseMask(ExecutionPhase::kDecode);
  constexpr uint8_t kKnownAliases =
      AliasMask(AliasMode::kDisjoint) | AliasMask(AliasMode::kExactLeft) |
      AliasMask(AliasMode::kExactRight) | AliasMask(AliasMode::kExactInPlace);
  constexpr uint32_t kKnownDtypes =
      (uint32_t{1} << (static_cast<uint8_t>(Dtype::kFloat64) + 1U)) - 1U;
  if (backend > BackendId::kCutlass || op > OpKind::kLogits || device_kind > DeviceKind::kCuda ||
      input_layout > LayoutId::kContiguousKvBshd || weight_layout > LayoutId::kContiguousKvBshd ||
      output_layout > LayoutId::kContiguousKvBshd || (phase_mask & ~kKnownPhases) != 0 ||
      (alias_mask & ~kKnownAliases) != 0 || (input_dtype_mask & ~kKnownDtypes) != 0 ||
      (weight_dtype_mask & ~kKnownDtypes) != 0 || (output_dtype_mask & ~kKnownDtypes) != 0) {
    return absl::InvalidArgumentError("kernel_capability.vocabulary: unknown enum or mask bit");
  }
  if (rank > KernelKey::kMaxDimensions) {
    return absl::InvalidArgumentError("kernel_capability.rank: exceeds key capacity");
  }
  for (size_t axis = 0; axis < rank; ++axis) {
    if (dimensions[axis].minimum > dimensions[axis].maximum) {
      return absl::InvalidArgumentError("kernel_capability.dimensions: invalid range");
    }
  }
  for (size_t axis = rank; axis < KernelKey::kMaxDimensions; ++axis) {
    if (dimensions[axis].minimum != 0 || dimensions[axis].maximum != 0) {
      return absl::InvalidArgumentError(
          "kernel_capability.dimensions: trailing dimensions must be zero");
    }
  }
  if ((device_kind == DeviceKind::kHost &&
       (minimum_compute_capability != 0 || maximum_compute_capability != 0)) ||
      (device_kind == DeviceKind::kCuda &&
       (minimum_compute_capability == 0 ||
        minimum_compute_capability > maximum_compute_capability))) {
    return absl::InvalidArgumentError("kernel_capability.compute_capability: invalid range");
  }
  if (minimum_operand_alignment == 0 ||
      (minimum_operand_alignment & (minimum_operand_alignment - 1U)) != 0 ||
      workspace_alignment == 0 || (workspace_alignment & (workspace_alignment - 1U)) != 0) {
    return absl::InvalidArgumentError("kernel_capability.alignment: must be powers of two");
  }
  if (head_dimension_multiple == 0) {
    return absl::InvalidArgumentError(
        "kernel_capability.head_dimension_multiple: must be positive");
  }
  if (launch_allocates) {
    return absl::InvalidArgumentError(
        "kernel_capability.launch: production launch may not allocate");
  }
  return absl::OkStatus();
}

absl::StatusOr<CapabilityMatch> BackendCapability::Match(const KernelKey& key) const {
  absl::Status status = Validate();
  if (!status.ok()) return status;
  status = ValidateKernelKey(key);
  if (!status.ok()) return status;
  auto reject = [&](const char* reason) -> absl::StatusOr<CapabilityMatch> {
    return absl::UnimplementedError(
        absl::StrCat("kernel_capability.", capability_id, ": ", reason));
  };
  if (key.op != op || key.device_kind != device_kind) return reject("operation/device mismatch");
  if ((phase_mask & PhaseMask(key.phase)) == 0) return reject("phase unsupported");
  if ((input_dtype_mask & DtypeMask(key.input_dtype)) == 0 ||
      (weight_dtype_mask & DtypeMask(key.weight_dtype)) == 0 ||
      (output_dtype_mask & DtypeMask(key.output_dtype)) == 0) {
    return reject("dtype unsupported");
  }
  if ((require_matching_input_weight_dtype && key.input_dtype != key.weight_dtype) ||
      (require_matching_weight_output_dtype && key.weight_dtype != key.output_dtype)) {
    return reject("dtype relationship unsupported");
  }
  if (key.input_layout != input_layout || key.weight_layout != weight_layout ||
      key.output_layout != output_layout || (alias_mask & AliasMask(key.alias_mode)) == 0) {
    return reject("layout or alias mode unsupported");
  }
  if (key.rank != rank) return reject("rank unsupported");
  for (size_t axis = 0; axis < rank; ++axis) {
    if (key.dimensions[axis] < dimensions[axis].minimum ||
        key.dimensions[axis] > dimensions[axis].maximum) {
      return reject("dimension outside capability range");
    }
  }
  if (device_kind == DeviceKind::kCuda && (key.compute_capability < minimum_compute_capability ||
                                           key.compute_capability > maximum_compute_capability)) {
    return reject("compute capability unsupported");
  }
  if (key.alignment_class < minimum_operand_alignment)
    return reject("operand alignment insufficient");
  if (key.op == OpKind::kAttention &&
      (key.query_heads > maximum_query_heads || key.kv_heads > maximum_kv_heads ||
       key.head_dimension > maximum_head_dimension ||
       key.head_dimension % head_dimension_multiple != 0 ||
       key.sequence_bucket > maximum_sequence_bucket)) {
    return reject("attention geometry outside capability range");
  }
  uint64_t output_elements = 1;
  for (size_t axis = 0; axis < rank; ++axis) {
    absl::StatusOr<uint64_t> next =
        CheckedMul(output_elements, key.dimensions[axis], "kernel_capability.workspace");
    if (!next.ok()) return next.status();
    output_elements = *next;
  }
  absl::StatusOr<uint64_t> variable = CheckedMul(
      output_elements, workspace_bytes_per_output_element, "kernel_capability.workspace");
  if (!variable.ok()) return variable.status();
  absl::StatusOr<uint64_t> workspace =
      CheckedAdd(fixed_workspace_bytes, *variable, "kernel_capability.workspace");
  if (!workspace.ok()) return workspace.status();
  if (*workspace > key.workspace_limit_bytes) return reject("workspace limit exceeded");
  return CapabilityMatch{*workspace, workspace_alignment};
}

KernelRegistry::KernelRegistry(size_t capacity) : capacity_(capacity) {
  entries_.reserve(capacity);
}

absl::Status KernelRegistry::Register(BackendCapability capability) {
  if (frozen_) return absl::FailedPreconditionError("kernel_registry: registry is frozen");
  absl::Status status = capability.Validate();
  if (!status.ok()) return status;
  if (entries_.size() == capacity_) {
    return absl::ResourceExhaustedError("kernel_registry: registration capacity exhausted");
  }
  for (const BackendCapability& existing : entries_) {
    if (existing.backend == capability.backend &&
        existing.capability_id == capability.capability_id) {
      return absl::AlreadyExistsError("kernel_registry: duplicate backend/capability ID");
    }
  }
  entries_.push_back(std::move(capability));
  return absl::OkStatus();
}

absl::Status KernelRegistry::Freeze() {
  if (frozen_) return absl::FailedPreconditionError("kernel_registry: registry is already frozen");
  std::sort(entries_.begin(), entries_.end(),
            [](const BackendCapability& left, const BackendCapability& right) {
              if (left.priority != right.priority) return left.priority < right.priority;
              if (left.backend != right.backend) return left.backend < right.backend;
              return left.capability_id < right.capability_id;
            });
  frozen_ = true;
  return absl::OkStatus();
}

absl::StatusOr<KernelSelection> KernelRegistry::Select(
    const KernelKey& key, std::optional<BackendId> forced_backend) const {
  if (!frozen_) return absl::FailedPreconditionError("kernel_registry: freeze before lookup");
  absl::Status status = ValidateKernelKey(key);
  if (!status.ok()) return status;
  if (forced_backend.has_value() && *forced_backend > BackendId::kCutlass) {
    return absl::InvalidArgumentError("kernel_registry: forced backend is outside vocabulary");
  }
  bool saw_forced_backend = false;
  absl::Status first_rejection;
  for (size_t index = 0; index < entries_.size(); ++index) {
    const BackendCapability& capability = entries_[index];
    if (forced_backend.has_value() && capability.backend != *forced_backend) continue;
    saw_forced_backend = true;
    absl::StatusOr<CapabilityMatch> match = capability.Match(key);
    if (match.ok()) {
      return KernelSelection{capability.backend, capability.capability_id, index, *match};
    }
    if (first_rejection.ok()) first_rejection = match.status();
  }
  if (forced_backend.has_value() && !saw_forced_backend) {
    return absl::UnimplementedError(absl::StrCat("kernel_registry: forced backend '",
                                                 BackendIdName(*forced_backend),
                                                 "' is not registered"));
  }
  if (!first_rejection.ok()) return first_rejection;
  return absl::UnimplementedError("kernel_registry: no matching capability");
}

const BackendCapability& KernelRegistry::capability(const KernelSelection& selection) const {
  return entries_[selection.registry_index];
}

PreparedKernelCache::PreparedKernelCache(size_t capacity) : capacity_(capacity) {
  plans_.reserve(capacity);
}

absl::Status PreparedKernelCache::Insert(PreparedKernelPlan plan) {
  if (frozen_) return absl::FailedPreconditionError("prepared_kernel_cache: cache is frozen");
  absl::Status status = ValidateKernelKey(plan.key);
  if (!status.ok()) return status;
  if (plan.backend > BackendId::kCutlass || plan.capability_id.empty() ||
      plan.workspace_alignment == 0 ||
      (plan.workspace_alignment & (plan.workspace_alignment - 1U)) != 0) {
    return absl::InvalidArgumentError("prepared_kernel_cache.plan: invalid capability/alignment");
  }
  if (plans_.size() == capacity_) {
    return absl::ResourceExhaustedError("prepared_kernel_cache: capacity exhausted");
  }
  if (Find(plan.key) != nullptr) {
    return absl::AlreadyExistsError("prepared_kernel_cache: duplicate key");
  }
  plans_.push_back(std::move(plan));
  return absl::OkStatus();
}

absl::Status PreparedKernelCache::Freeze() {
  if (frozen_) return absl::FailedPreconditionError("prepared_kernel_cache: cache already frozen");
  std::sort(plans_.begin(), plans_.end(),
            [](const PreparedKernelPlan& left, const PreparedKernelPlan& right) {
              return KernelKeyLess(left.key, right.key);
            });
  frozen_ = true;
  return absl::OkStatus();
}

const PreparedKernelPlan* PreparedKernelCache::Find(const KernelKey& key) const noexcept {
  for (const PreparedKernelPlan& plan : plans_) {
    if (plan.key == key) return &plan;
  }
  return nullptr;
}

}  // namespace inferx::ops
