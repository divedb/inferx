#include "inferx/runtime/memory_tracker.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"

namespace inferx {
namespace {

absl::Status ValidateCategory(MemoryCategory category) {
  switch (category) {
    case MemoryCategory::kModelWeights:
    case MemoryCategory::kKvCache:
    case MemoryCategory::kWorkspace:
    case MemoryCategory::kExecutionMetadata:
    case MemoryCategory::kPinnedStaging:
    case MemoryCategory::kRuntimeInternal:
    case MemoryCategory::kTest:
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError("memory.category: invalid category");
}

class TrackerAccountingReservation final : public AllocationAccountingReservation {
 public:
  explicit TrackerAccountingReservation(MemoryReservation reservation)
      : reservation_(std::move(reservation)) {}

  absl::Status Commit() override {
    if (!reservation_.has_value()) {
      return absl::FailedPreconditionError("memory.accounting.commit: reservation is inactive");
    }
    absl::StatusOr<AllocationCharge> charge = reservation_->Commit();
    if (!charge.ok()) return charge.status();
    charge_.emplace(std::move(*charge));
    reservation_.reset();
    return absl::OkStatus();
  }

  absl::Status Release() override {
    if (!charge_.has_value()) {
      return absl::FailedPreconditionError("memory.accounting.release: charge is inactive");
    }
    absl::Status status = charge_->Release();
    if (status.ok()) charge_.reset();
    return status;
  }

  void Abandon() noexcept override {
    if (charge_.has_value()) {
      static_cast<void>(charge_->Release());
      charge_.reset();
    }
    reservation_.reset();
  }

 private:
  std::optional<MemoryReservation> reservation_;
  std::optional<AllocationCharge> charge_;
};

}  // namespace

AllocationCharge::AllocationCharge(MemoryTracker* tracker, MemoryKey key, ByteCount bytes) noexcept
    : tracker_(tracker), key_(key), bytes_(bytes) {}

AllocationCharge::~AllocationCharge() noexcept { Abandon(); }

AllocationCharge::AllocationCharge(AllocationCharge&& other) noexcept
    : tracker_(std::exchange(other.tracker_, nullptr)),
      key_(other.key_),
      bytes_(std::exchange(other.bytes_, ByteCount(0))) {}

AllocationCharge& AllocationCharge::operator=(AllocationCharge&& other) noexcept {
  if (this != &other) {
    Abandon();
    tracker_ = std::exchange(other.tracker_, nullptr);
    key_ = other.key_;
    bytes_ = std::exchange(other.bytes_, ByteCount(0));
  }
  return *this;
}

absl::Status AllocationCharge::Release() {
  if (tracker_ == nullptr) {
    return absl::FailedPreconditionError("memory.charge: charge is inactive");
  }
  absl::Status status = tracker_->Release(key_, bytes_);
  if (status.ok()) {
    tracker_ = nullptr;
    bytes_ = ByteCount(0);
  }
  return status;
}

absl::Status AllocationCharge::MarkLeaked() {
  if (tracker_ == nullptr) {
    return absl::FailedPreconditionError("memory.charge: charge is inactive");
  }
  absl::Status status = tracker_->MarkLeaked(key_, bytes_);
  if (status.ok()) {
    tracker_ = nullptr;
    bytes_ = ByteCount(0);
  }
  return status;
}

void AllocationCharge::Abandon() noexcept {
  if (tracker_ != nullptr) {
    static_cast<void>(tracker_->MarkLeaked(key_, bytes_));
    tracker_ = nullptr;
    bytes_ = ByteCount(0);
  }
}

MemoryReservation::MemoryReservation(MemoryTracker* tracker, MemoryKey key,
                                     ByteCount bytes) noexcept
    : tracker_(tracker), key_(key), bytes_(bytes) {}

MemoryReservation::~MemoryReservation() noexcept { Abandon(); }

MemoryReservation::MemoryReservation(MemoryReservation&& other) noexcept
    : tracker_(std::exchange(other.tracker_, nullptr)),
      key_(other.key_),
      bytes_(std::exchange(other.bytes_, ByteCount(0))) {}

MemoryReservation& MemoryReservation::operator=(MemoryReservation&& other) noexcept {
  if (this != &other) {
    Abandon();
    tracker_ = std::exchange(other.tracker_, nullptr);
    key_ = other.key_;
    bytes_ = std::exchange(other.bytes_, ByteCount(0));
  }
  return *this;
}

absl::StatusOr<AllocationCharge> MemoryReservation::Commit() {
  if (tracker_ == nullptr) {
    return absl::FailedPreconditionError("memory.reservation: reservation is inactive");
  }
  absl::Status status = tracker_->Commit(key_, bytes_);
  if (!status.ok()) {
    return status;
  }
  AllocationCharge result(tracker_, key_, bytes_);
  tracker_ = nullptr;
  bytes_ = ByteCount(0);
  return result;
}

absl::Status MemoryReservation::Rollback() {
  if (tracker_ == nullptr) {
    return absl::FailedPreconditionError("memory.reservation: reservation is inactive");
  }
  absl::Status status = tracker_->Rollback(key_, bytes_);
  if (status.ok()) {
    tracker_ = nullptr;
    bytes_ = ByteCount(0);
  }
  return status;
}

void MemoryReservation::Abandon() noexcept {
  if (tracker_ != nullptr) {
    static_cast<void>(tracker_->Rollback(key_, bytes_));
    tracker_ = nullptr;
    bytes_ = ByteCount(0);
  }
}

absl::StatusOr<MemoryTracker> MemoryTracker::Create(std::span<const MemoryLimit> limits) {
  MemoryTracker result;
  for (const MemoryLimit& item : limits) {
    absl::Status status = ValidateMemoryKind(item.device, item.kind);
    if (!status.ok()) {
      return status;
    }
    DomainKey key{item.device, item.kind};
    if (result.limits_.contains(key)) {
      return absl::InvalidArgumentError("memory.limits: duplicate device/memory-kind limit");
    }
    result.limits_.emplace(key, item.limit);
  }
  return result;
}

MemoryTracker::MemoryTracker(MemoryTracker&& other) noexcept {
  std::lock_guard<std::mutex> lock(other.mutex_);
  limits_ = std::move(other.limits_);
  counters_ = std::move(other.counters_);
}

uint64_t MemoryTracker::DomainUsageLocked(DomainKey domain) const {
  uint64_t usage = 0;
  for (const auto& [key, counters] : counters_) {
    if (key.device == domain.device && key.kind == domain.kind) {
      for (const uint64_t value :
           {counters.reserved.value(), counters.committed.value(), counters.leaked.value()}) {
        if (value > std::numeric_limits<uint64_t>::max() - usage) {
          return std::numeric_limits<uint64_t>::max();
        }
        usage += value;
      }
    }
  }
  return usage;
}

absl::StatusOr<MemoryReservation> MemoryTracker::BeginReservation(MemoryKey key, ByteCount bytes) {
  absl::Status status = ValidateMemoryKind(key.device, key.kind);
  if (!status.ok()) {
    return status;
  }
  status = ValidateCategory(key.category);
  if (!status.ok()) {
    return status;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const DomainKey domain{key.device, key.kind};
  const auto limit = limits_.find(domain);
  if (limit == limits_.end()) {
    return absl::NotFoundError("memory.limit: no limit configured for device/memory kind");
  }
  const uint64_t usage = DomainUsageLocked(domain);
  if (bytes.value() > limit->second.value() || usage > limit->second.value() - bytes.value()) {
    return absl::ResourceExhaustedError("memory.limit: reservation exceeds configured budget");
  }
  MemoryCounters& counters = counters_[key];
  counters.limit = limit->second;
  counters.reserved = ByteCount(counters.reserved.value() + bytes.value());
  return MemoryReservation(this, key, bytes);
}

absl::StatusOr<std::unique_ptr<AllocationAccountingReservation>>
MemoryTracker::BeginAllocationReservation(const AllocationRequest& request) {
  absl::StatusOr<MemoryReservation> reservation = BeginReservation(
      MemoryKey{request.device, request.memory_kind, request.category}, request.bytes);
  if (!reservation.ok()) return reservation.status();
  try {
    return std::unique_ptr<AllocationAccountingReservation>(
        new TrackerAccountingReservation(std::move(*reservation)));
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("memory.accounting: reservation-handle allocation failed");
  }
}

absl::Status MemoryTracker::Commit(MemoryKey key, ByteCount bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto item = counters_.find(key);
  if (item == counters_.end() || item->second.reserved.value() < bytes.value()) {
    return absl::InternalError("memory.commit: reservation accounting mismatch");
  }
  MemoryCounters& counters = item->second;
  if (counters.live_allocations == std::numeric_limits<uint64_t>::max()) {
    return absl::OutOfRangeError("memory.live_allocations: counter overflow");
  }
  counters.reserved = ByteCount(counters.reserved.value() - bytes.value());
  counters.committed = ByteCount(counters.committed.value() + bytes.value());
  counters.peak_committed =
      ByteCount(std::max(counters.peak_committed.value(), counters.committed.value()));
  ++counters.live_allocations;
  return absl::OkStatus();
}

absl::Status MemoryTracker::Rollback(MemoryKey key, ByteCount bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto item = counters_.find(key);
  if (item == counters_.end() || item->second.reserved.value() < bytes.value()) {
    return absl::InternalError("memory.rollback: reservation accounting mismatch");
  }
  item->second.reserved = ByteCount(item->second.reserved.value() - bytes.value());
  return absl::OkStatus();
}

absl::Status MemoryTracker::Release(MemoryKey key, ByteCount bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto item = counters_.find(key);
  if (item == counters_.end() || item->second.committed.value() < bytes.value() ||
      item->second.live_allocations == 0) {
    return absl::InternalError("memory.release: charge accounting mismatch");
  }
  item->second.committed = ByteCount(item->second.committed.value() - bytes.value());
  --item->second.live_allocations;
  return absl::OkStatus();
}

absl::Status MemoryTracker::MarkLeaked(MemoryKey key, ByteCount bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto item = counters_.find(key);
  if (item == counters_.end() || item->second.committed.value() < bytes.value() ||
      item->second.live_allocations == 0) {
    return absl::InternalError("memory.leak: charge accounting mismatch");
  }
  item->second.committed = ByteCount(item->second.committed.value() - bytes.value());
  item->second.leaked = ByteCount(item->second.leaked.value() + bytes.value());
  return absl::OkStatus();
}

std::vector<std::pair<MemoryKey, MemoryCounters>> MemoryTracker::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {counters_.begin(), counters_.end()};
}

absl::Status MemoryTracker::ValidateBaseline(bool allow_leaks) const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& [key, counters] : counters_) {
    static_cast<void>(key);
    if (counters.reserved.value() != 0 || counters.committed.value() != 0) {
      return absl::FailedPreconditionError("memory.baseline: live reservation or charge remains");
    }
    if (!allow_leaks && (counters.leaked.value() != 0 || counters.live_allocations != 0)) {
      return absl::FailedPreconditionError("memory.baseline: leaked allocation remains");
    }
  }
  return absl::OkStatus();
}

}  // namespace inferx
