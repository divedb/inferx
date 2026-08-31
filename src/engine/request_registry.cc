#include "inferx/engine/request_registry.h"

#include <limits>
#include <new>

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
  if (next_arrival_ordinal_ == std::numeric_limits<uint64_t>::max()) {
    return WithErrorReason(absl::OutOfRangeError("registry.sequence: id space exhausted"),
                           ErrorReason::kInvariantViolation);
  }
  // The registry owns the ordinal/sequence assignment; the caller supplies
  // only the arrival time.
  context->arrival = ArrivalKey{context->arrival.arrival_time, next_arrival_ordinal_, id};
  context->sequence = SequenceId(next_arrival_ordinal_ + 1);
  const ArrivalKey key = context->arrival;

  // Transactional across both containers and the ID counter. Allocation
  // exceptions are translated at this module boundary.
  try {
    const auto [inserted, fresh] = by_id_.emplace(id, std::move(context));
    if (!fresh) {
      return WithErrorReason(absl::AlreadyExistsError("registry.insert: duplicate request id"),
                             ErrorReason::kDuplicateRequest);
    }
    arrival_order_.emplace(key, id);
    ++next_arrival_ordinal_;
    static_cast<void>(inserted);
  } catch (const std::bad_alloc&) {
    by_id_.erase(id);
    return WithErrorReason(absl::ResourceExhaustedError("registry.insert: allocation failed"),
                           ErrorReason::kQueueFull);
  } catch (...) {
    by_id_.erase(id);
    return WithErrorReason(absl::InternalError("registry.insert: unexpected exception"),
                           ErrorReason::kInvariantViolation);
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

void RequestRegistry::AppendSchedulingViews(
    std::vector<scheduler::RequestSchedulingView>& output) const {
  output.reserve(output.size() + arrival_order_.size());
  for (const auto& [key, id] : arrival_order_) {
    static_cast<void>(key);
    const RequestContext* context = Find(id);
    if (context == nullptr) {
      continue;
    }
    output.push_back(scheduler::RequestSchedulingView{
        .request = context->request->id,
        .sequence = context->sequence,
        .epoch = context->epoch,
        .model = context->request->model,
        .state = context->state,
        .arrival_ordinal = context->arrival.arrival_ordinal,
        .arrival_time = context->arrival.arrival_time,
        .prompt_tokens = TokenCount(static_cast<uint32_t>(context->prompt_tokens.size())),
        .computed_tokens = TokenCount(context->num_computed_tokens),
        .committed_output_tokens = TokenCount(context->num_committed_output_tokens),
        .max_output_tokens = context->reservation_output_tokens.value() == 0
                                 ? context->request->generation.max_output_tokens
                                 : context->reservation_output_tokens,
        .deadline = context->request->deadline,
        .reservation = context->reservation,
    });
  }
}

}  // namespace inferx
