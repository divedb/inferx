#include "inferx/scheduler/resource_accountant.h"

#include <cassert>
#include <limits>
#include <new>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"

namespace inferx::scheduler {
namespace {

absl::Status ResourceError(absl::StatusCode code, absl::string_view message, ErrorReason reason) {
  return WithErrorReason(absl::Status(code, absl::StrCat("scheduler.resources: ", message)),
                         reason);
}

}  // namespace

ReservationTxn::ReservationTxn(ResourceAccountant& owner, ReservationId id,
                               ResourceCost cost) noexcept
    : owner_(&owner), id_(id), cost_(cost) {}

ReservationTxn::ReservationTxn(ReservationTxn&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), id_(other.id_), cost_(other.cost_) {}

ReservationTxn& ReservationTxn::operator=(ReservationTxn&& other) noexcept {
  if (this != &other) {
    Rollback();
    owner_ = std::exchange(other.owner_, nullptr);
    id_ = other.id_;
    cost_ = other.cost_;
  }
  return *this;
}

ReservationTxn::~ReservationTxn() noexcept { Rollback(); }

ReservationId ReservationTxn::Commit() noexcept {
  assert(owner_ != nullptr);
  ResourceAccountant* owner = std::exchange(owner_, nullptr);
  return owner->CommitTentative(id_);
}

void ReservationTxn::Rollback() noexcept {
  if (owner_ != nullptr) {
    owner_->RollbackTentative(id_);
    owner_ = nullptr;
  }
}

ResourceAccountant::ResourceAccountant(SequenceCount sequence_capacity,
                                       KvTokenCount kv_token_capacity) noexcept
    : resources_{sequence_capacity, SequenceCount(0), kv_token_capacity, KvTokenCount(0)} {}

ResourceSnapshot ResourceAccountant::Snapshot() const noexcept { return resources_; }

absl::StatusOr<ReservationTxn> ResourceAccountant::BeginReservation(RequestId request,
                                                                    ResourceCost cost) {
  if (tentative_.has_value()) {
    return ResourceError(absl::StatusCode::kFailedPrecondition,
                         "another tentative reservation is already live",
                         ErrorReason::kInvariantViolation);
  }
  if (cost.sequences.value() == 0 || cost.kv_tokens.value() == 0) {
    return ResourceError(absl::StatusCode::kInvalidArgument,
                         "reservation cost must contain positive sequences and KV tokens",
                         ErrorReason::kInvalidRequest);
  }
  for (const auto& [id, record] : reservations_) {
    static_cast<void>(id);
    if (record.request == request) {
      return ResourceError(absl::StatusCode::kAlreadyExists,
                           absl::StrCat("request ", request.value(), " already owns capacity"),
                           ErrorReason::kUnknownResource);
    }
  }
  if (next_reservation_id_ == std::numeric_limits<uint64_t>::max()) {
    return ResourceError(absl::StatusCode::kOutOfRange, "reservation id exhausted",
                         ErrorReason::kInvariantViolation);
  }

  absl::StatusOr<uint32_t> new_sequences = CheckedAdd(
      resources_.sequences_used.value(), cost.sequences.value(), "scheduler.resources.sequences");
  if (!new_sequences.ok()) {
    return new_sequences.status();
  }
  absl::StatusOr<uint64_t> new_kv =
      CheckedAdd(resources_.kv_tokens_used.value(), static_cast<uint64_t>(cost.kv_tokens.value()),
                 "scheduler.resources.kv_tokens");
  if (!new_kv.ok()) {
    return new_kv.status();
  }
  if (*new_sequences > resources_.sequence_capacity.value() ||
      *new_kv > resources_.kv_token_capacity.value()) {
    return ResourceError(absl::StatusCode::kResourceExhausted,
                         absl::StrCat("capacity unavailable for request ", request.value()),
                         ErrorReason::kCapacityExhausted);
  }

  const ReservationId id(next_reservation_id_);
  try {
    const auto [position, inserted] =
        reservations_.emplace(id, Record{request, cost, *new_sequences, *new_kv, false});
    static_cast<void>(position);
    if (!inserted) {
      return ResourceError(absl::StatusCode::kInternal, "provisional id collision",
                           ErrorReason::kInvariantViolation);
    }
  } catch (const std::bad_alloc&) {
    return ResourceError(absl::StatusCode::kResourceExhausted,
                         "allocation failed while preparing reservation",
                         ErrorReason::kCapacityExhausted);
  } catch (...) {
    return ResourceError(absl::StatusCode::kInternal,
                         "unexpected exception while preparing reservation",
                         ErrorReason::kInvariantViolation);
  }
  tentative_ = id;
  return ReservationTxn(*this, id, cost);
}

absl::Status ResourceAccountant::Release(ReservationId id) {
  const auto found = reservations_.find(id);
  if (found == reservations_.end() || !found->second.committed) {
    return ResourceError(absl::StatusCode::kNotFound,
                         absl::StrCat("reservation ", id.value(), " is unknown"),
                         ErrorReason::kUnknownResource);
  }
  absl::StatusOr<uint32_t> new_sequences =
      CheckedSub(resources_.sequences_used.value(), found->second.cost.sequences.value(),
                 "scheduler.resources.release.sequences");
  absl::StatusOr<uint64_t> new_kv =
      CheckedSub(resources_.kv_tokens_used.value(),
                 static_cast<uint64_t>(found->second.cost.kv_tokens.value()),
                 "scheduler.resources.release.kv_tokens");
  if (!new_sequences.ok() || !new_kv.ok()) {
    return ResourceError(absl::StatusCode::kInternal, "committed counters underflow on release",
                         ErrorReason::kInvariantViolation);
  }
  resources_.sequences_used = SequenceCount(*new_sequences);
  resources_.kv_tokens_used = KvTokenCount(*new_kv);
  reservations_.erase(found);
  return absl::OkStatus();
}

absl::Status ResourceAccountant::Validate() const {
  uint32_t sequences = 0;
  uint64_t kv_tokens = 0;
  size_t tentative_count = 0;
  for (const auto& [id, record] : reservations_) {
    if (!record.committed) {
      ++tentative_count;
      if (!tentative_.has_value() || *tentative_ != id) {
        return ResourceError(absl::StatusCode::kInternal, "orphan tentative record",
                             ErrorReason::kInvariantViolation);
      }
      continue;
    }
    absl::StatusOr<uint32_t> next_sequences = CheckedAdd(sequences, record.cost.sequences.value(),
                                                         "scheduler.resources.validate.sequences");
    absl::StatusOr<uint64_t> next_kv =
        CheckedAdd(kv_tokens, static_cast<uint64_t>(record.cost.kv_tokens.value()),
                   "scheduler.resources.validate.kv_tokens");
    if (!next_sequences.ok() || !next_kv.ok()) {
      return ResourceError(absl::StatusCode::kInternal, "record sum overflow",
                           ErrorReason::kInvariantViolation);
    }
    sequences = *next_sequences;
    kv_tokens = *next_kv;
  }
  if (tentative_count > 1 || tentative_count != (tentative_.has_value() ? 1U : 0U) ||
      sequences != resources_.sequences_used.value() ||
      kv_tokens != resources_.kv_tokens_used.value() ||
      sequences > resources_.sequence_capacity.value() ||
      kv_tokens > resources_.kv_token_capacity.value()) {
    return ResourceError(absl::StatusCode::kInternal, "accounting invariant mismatch",
                         ErrorReason::kInvariantViolation);
  }
  return absl::OkStatus();
}

bool ResourceAccountant::HasReservation(ReservationId id) const {
  const auto found = reservations_.find(id);
  return found != reservations_.end() && found->second.committed;
}

std::optional<ResourceCost> ResourceAccountant::LookupCost(ReservationId id) const {
  const auto found = reservations_.find(id);
  if (found == reservations_.end() || !found->second.committed) {
    return std::nullopt;
  }
  return found->second.cost;
}

size_t ResourceAccountant::live_reservations() const noexcept {
  return reservations_.size() - (tentative_.has_value() ? 1U : 0U);
}

ReservationId ResourceAccountant::CommitTentative(ReservationId id) noexcept {
  assert(tentative_.has_value() && *tentative_ == id);
  auto found = reservations_.find(id);
  assert(found != reservations_.end() && !found->second.committed);
  resources_.sequences_used = SequenceCount(found->second.prepared_sequences_used);
  resources_.kv_tokens_used = KvTokenCount(found->second.prepared_kv_tokens_used);
  found->second.committed = true;
  tentative_.reset();
  ++next_reservation_id_;
  return id;
}

void ResourceAccountant::RollbackTentative(ReservationId id) noexcept {
  assert(tentative_.has_value() && *tentative_ == id);
  reservations_.erase(id);
  tentative_.reset();
}

}  // namespace inferx::scheduler
