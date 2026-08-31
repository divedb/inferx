#include "inferx/scheduler/admission_controller.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"

namespace inferx::scheduler {
namespace {

AdmissionDecision NeverFits(const RequestSchedulingView& request, absl::string_view detail) {
  return AdmissionDecision{
      .kind = AdmissionKind::kNeverFits,
      .cost = ResourceCost{},
      .rejection = WithErrorReason(
          absl::Status(
              absl::StatusCode::kResourceExhausted,
              absl::StrCat("scheduler.admission.request.", request.request.value(), ": ", detail)),
          ErrorReason::kImpossibleContext),
  };
}

}  // namespace

AdmissionDecision AdmissionController::Evaluate(const RequestSchedulingView& request,
                                                const ResourceSnapshot& resources) const {
  absl::StatusOr<uint32_t> kv_cost =
      CheckedAdd(request.prompt_tokens.value(), request.max_output_tokens.value(),
                 "scheduler.admission.kv_cost");
  if (!kv_cost.ok()) {
    return NeverFits(request, "prompt and output reservation overflows");
  }
  const ResourceCost cost{SequenceCount(1), TokenCount(*kv_cost)};
  if (cost.sequences.value() > resources.sequence_capacity.value() ||
      static_cast<uint64_t>(cost.kv_tokens.value()) > resources.kv_token_capacity.value()) {
    AdmissionDecision decision = NeverFits(request, "full reservation can never fit capacity");
    decision.cost = cost;
    return decision;
  }

  absl::StatusOr<uint32_t> used_sequences = CheckedAdd(
      resources.sequences_used.value(), cost.sequences.value(), "scheduler.admission.sequences");
  absl::StatusOr<uint64_t> used_kv =
      CheckedAdd(resources.kv_tokens_used.value(), static_cast<uint64_t>(cost.kv_tokens.value()),
                 "scheduler.admission.kv_tokens");
  if (!used_sequences.ok() || !used_kv.ok() ||
      (used_sequences.ok() && *used_sequences > resources.sequence_capacity.value()) ||
      (used_kv.ok() && *used_kv > resources.kv_token_capacity.value())) {
    return AdmissionDecision{AdmissionKind::kTemporarilyBlocked, cost, absl::OkStatus()};
  }
  return AdmissionDecision{AdmissionKind::kReservable, cost, absl::OkStatus()};
}

}  // namespace inferx::scheduler
