// Transactional, thread-safe memory budget accounting.
#ifndef INFERX_RUNTIME_MEMORY_TRACKER_H_
#define INFERX_RUNTIME_MEMORY_TRACKER_H_

#include <cstdint>
#include <map>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/token.h"
#include "inferx/runtime/memory_category.h"
#include "inferx/tensor/device.h"

namespace inferx {

struct MemoryKey {
  Device device;
  MemoryKind kind;
  MemoryCategory category;

  friend constexpr bool operator==(const MemoryKey&, const MemoryKey&) = default;
  friend constexpr bool operator<(const MemoryKey& left, const MemoryKey& right) {
    if (left.device < right.device) return true;
    if (right.device < left.device) return false;
    if (left.kind != right.kind) return left.kind < right.kind;
    return left.category < right.category;
  }
};

struct MemoryLimit {
  Device device;
  MemoryKind kind;
  ByteCount limit;
};

struct MemoryCounters {
  ByteCount limit = ByteCount(0);
  ByteCount reserved = ByteCount(0);
  ByteCount committed = ByteCount(0);
  ByteCount peak_committed = ByteCount(0);
  ByteCount leaked = ByteCount(0);
  uint64_t live_allocations = 0;
};

class MemoryTracker;

class AllocationCharge {
 public:
  AllocationCharge() noexcept = default;
  ~AllocationCharge() noexcept;
  AllocationCharge(AllocationCharge&& other) noexcept;
  AllocationCharge& operator=(AllocationCharge&& other) noexcept;
  AllocationCharge(const AllocationCharge&) = delete;
  AllocationCharge& operator=(const AllocationCharge&) = delete;

  [[nodiscard]] bool active() const noexcept { return tracker_ != nullptr; }
  [[nodiscard]] ByteCount bytes() const noexcept { return bytes_; }
  [[nodiscard]] absl::Status Release();
  [[nodiscard]] absl::Status MarkLeaked();

 private:
  friend class MemoryTracker;
  friend class MemoryReservation;
  AllocationCharge(MemoryTracker* tracker, MemoryKey key, ByteCount bytes) noexcept;
  void Abandon() noexcept;

  MemoryTracker* tracker_ = nullptr;
  MemoryKey key_{Device::Host(), MemoryKind::kHost, MemoryCategory::kRuntimeInternal};
  ByteCount bytes_ = ByteCount(0);
};

class MemoryReservation {
 public:
  MemoryReservation() noexcept = default;
  ~MemoryReservation() noexcept;
  MemoryReservation(MemoryReservation&& other) noexcept;
  MemoryReservation& operator=(MemoryReservation&& other) noexcept;
  MemoryReservation(const MemoryReservation&) = delete;
  MemoryReservation& operator=(const MemoryReservation&) = delete;

  [[nodiscard]] bool active() const noexcept { return tracker_ != nullptr; }
  [[nodiscard]] absl::StatusOr<AllocationCharge> Commit();
  [[nodiscard]] absl::Status Rollback();

 private:
  friend class MemoryTracker;
  MemoryReservation(MemoryTracker* tracker, MemoryKey key, ByteCount bytes) noexcept;
  void Abandon() noexcept;

  MemoryTracker* tracker_ = nullptr;
  MemoryKey key_{Device::Host(), MemoryKind::kHost, MemoryCategory::kRuntimeInternal};
  ByteCount bytes_ = ByteCount(0);
};

class MemoryTracker : public AllocationAccounting {
 public:
  [[nodiscard]] static absl::StatusOr<MemoryTracker> Create(std::span<const MemoryLimit> limits);

  ~MemoryTracker() noexcept;
  MemoryTracker(MemoryTracker&& other) noexcept;
  MemoryTracker& operator=(MemoryTracker&&) = delete;
  MemoryTracker(const MemoryTracker&) = delete;
  MemoryTracker& operator=(const MemoryTracker&) = delete;

  [[nodiscard]] absl::StatusOr<MemoryReservation> BeginReservation(MemoryKey key, ByteCount bytes);
  [[nodiscard]] absl::StatusOr<std::unique_ptr<AllocationAccountingReservation>>
  BeginAllocationReservation(const AllocationRequest& request) override;
  [[nodiscard]] std::vector<std::pair<MemoryKey, MemoryCounters>> Snapshot() const;
  [[nodiscard]] absl::Status ValidateBaseline(bool allow_leaks = false) const;

 private:
  friend class MemoryReservation;
  friend class AllocationCharge;

  struct DomainKey {
    Device device;
    MemoryKind kind;
    friend constexpr bool operator<(const DomainKey& left, const DomainKey& right) {
      if (left.device < right.device) return true;
      if (right.device < left.device) return false;
      return left.kind < right.kind;
    }
  };

  MemoryTracker() = default;
  [[nodiscard]] absl::Status Commit(MemoryKey key, ByteCount bytes);
  [[nodiscard]] absl::Status Rollback(MemoryKey key, ByteCount bytes);
  [[nodiscard]] absl::Status Release(MemoryKey key, ByteCount bytes);
  [[nodiscard]] absl::Status MarkLeaked(MemoryKey key, ByteCount bytes);
  [[nodiscard]] uint64_t DomainUsageLocked(DomainKey domain) const;

  mutable std::mutex mutex_;
  std::map<DomainKey, ByteCount> limits_;
  std::map<MemoryKey, MemoryCounters> counters_;
};

}  // namespace inferx

#endif  // INFERX_RUNTIME_MEMORY_TRACKER_H_
