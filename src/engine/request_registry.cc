#include "inferx/engine/request_registry.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/status.h"

namespace inferx {

absl::Status RequestRegistry::Insert(std::unique_ptr<RequestContext> context) {
  if (context == nullptr) {
    return absl::InvalidArgumentError("registry.insert: null context");
  }
  const RequestId id = context->request->id;
  if (by_id_.find(id) != by_id_.end()) {
    return WithErrorReason(
        absl::Status(absl::StatusCode::kAlreadyExists,
                     absl::StrCat("registry.request.", id.value(), ": duplicate request id")),
        ErrorReason::kDuplicateRequest);
  }
  // The registry owns the ordinal/sequence assignment; the caller supplies
  // only the arrival time.
  context->arrival = ArrivalKey{context->arrival.arrival_time, next_arrival_ordinal_, id};
  context->sequence = SequenceId(next_arrival_ordinal_ + 1);
  ++next_arrival_ordinal_;
  const ArrivalKey key = context->arrival;

  // Transactional across both containers: roll back the first insertion if
  // the second throws.
  by_id_.emplace(id, std::move(context));
  try {
    arrival_order_.emplace(key, id);
  } catch (...) {
    by_id_.erase(id);
    throw;
  }
  return absl::OkStatus();
}

RequestContext* RequestRegistry::Find(RequestId id) {
  auto found = by_id_.find(id);
  return found == by_id_.end() ? nullptr : found->second.get();
}

const RequestContext* RequestRegistry::Find(RequestId id) const {
  auto found = by_id_.find(id);
  return found == by_id_.end() ? nullptr : found->second.get();
}

bool RequestRegistry::Contains(RequestId id) const { return by_id_.contains(id); }

absl::Status RequestRegistry::EraseTerminal(RequestId id) {
  auto found = by_id_.find(id);
  if (found == by_id_.end()) {
    return WithErrorReason(absl::Status(absl::StatusCode::kNotFound,
                                        absl::StrCat("registry.request.", id.value(), ": unknown")),
                           ErrorReason::kUnknownResource);
  }
  RequestContext& context = *found->second;
  if (!context.terminal_emitted || context.in_flight_ticket.has_value() ||
      context.reservation.has_value()) {
    return WithErrorReason(absl::Status(absl::StatusCode::kFailedPrecondition,
                                        absl::StrCat("registry.request.", id.value(),
                                                     ": erase requires emitted terminal, no "
                                                     "in-flight ticket, released reservation")),
                           ErrorReason::kInvariantViolation);
  }
  arrival_order_.erase(context.arrival);
  by_id_.erase(found);
  return absl::OkStatus();
}

void RequestRegistry::AppendArrivalOrder(std::vector<const RequestContext*>& output) const {
  output.reserve(output.size() + arrival_order_.size());
  for (const auto& [key, id] : arrival_order_) {
    const RequestContext* context = Find(id);
    if (context != nullptr) {
      output.push_back(context);
    }
  }
}

}  // namespace inferx
