// Coordinator-owned simulated capacity accounting.

#ifndef INFERX_SCHEDULER_RESOURCE_ACCOUNTANT_H_
#define INFERX_SCHEDULER_RESOURCE_ACCOUNTANT_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"

namespace inferx::scheduler {

struct ResourceCost {
  SequenceCount sequences{0};
  TokenCount kv_tokens{0};

  friend bool operator==(const ResourceCost&, const ResourceCost&) = default;
};

struct ResourceSnapshot {
  SequenceCount sequence_capacity{0};
  SequenceCount sequences_used{0};
  KvTokenCount kv_token_capacity{0};
  KvTokenCount kv_tokens_used{0};

  friend bool operator==(const ResourceSnapshot&, const ResourceSnapshot&) = default;
};

class ResourceAccountant;

// A tentative reservation has exactly one owner. Destruction before Commit
// removes the prepared record without changing committed counters.
class ReservationTxn {
 public:
  ReservationTxn(ReservationTxn&& other) noexcept;
  ReservationTxn& operator=(ReservationTxn&& other) noexcept;
  ReservationTxn(const ReservationTxn&) = delete;
  ReservationTxn& operator=(const ReservationTxn&) = delete;
  ~ReservationTxn() noexcept;

  [[nodiscard]] ReservationId id() const noexcept { return id_; }
  [[nodiscard]] ResourceCost cost() const noexcept { return cost_; }
  [[nodiscard]] ReservationId Commit() noexcept;

 private:
  friend class ResourceAccountant;
  ReservationTxn(ResourceAccountant& owner, ReservationId id, ResourceCost cost) noexcept;
  void Rollback() noexcept;

  ResourceAccountant* owner_ = nullptr;
  ReservationId id_{0};
  ResourceCost cost_{};
};

class ResourceAccountant {
 public:
  ResourceAccountant(SequenceCount sequence_capacity, KvTokenCount kv_token_capacity) noexcept;

  [[nodiscard]] ResourceSnapshot Snapshot() const noexcept;
  [[nodiscard]] absl::StatusOr<ReservationTxn> BeginReservation(RequestId request,
                                                                ResourceCost cost);
  [[nodiscard]] absl::Status Release(ReservationId id);
  [[nodiscard]] absl::Status Validate() const;

  [[nodiscard]] bool HasReservation(ReservationId id) const;
  [[nodiscard]] std::optional<ResourceCost> LookupCost(ReservationId id) const;
  [[nodiscard]] size_t live_reservations() const noexcept;
  [[nodiscard]] bool has_tentative() const noexcept { return tentative_.has_value(); }

 private:
  friend class ReservationTxn;

  struct Record {
    RequestId request;
    ResourceCost cost;
    uint32_t prepared_sequences_used = 0;
    uint64_t prepared_kv_tokens_used = 0;
    bool committed = false;
  };

  [[nodiscard]] ReservationId CommitTentative(ReservationId id) noexcept;
  void RollbackTentative(ReservationId id) noexcept;

  ResourceSnapshot resources_;
  std::map<ReservationId, Record> reservations_;
  std::optional<ReservationId> tentative_;
  uint64_t next_reservation_id_ = 1;
};

}  // namespace inferx::scheduler

#endif  // INFERX_SCHEDULER_RESOURCE_ACCOUNTANT_H_
