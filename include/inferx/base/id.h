// Strong identity types (m1.md section 7.2; ADR 0008).
//
// IDs are explicit, non-default-constructible, and non-interchangeable.
// Absence is std::optional<Id>; zero is a valid value wherever the
// representation allows it — there is deliberately no numeric sentinel.

#ifndef INFERX_BASE_ID_H_
#define INFERX_BASE_ID_H_

#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/has_absl_stringify.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace inferx {

// Tag types give each ID a distinct identity; representations follow the
// m1.md section 7.2 table.
struct RequestIdTag {};
struct SequenceIdTag {};
struct StepIdTag {};
struct ExecutionTicketIdTag {};
struct ReservationIdTag {};
struct RequestEpochTag {};
struct BlockGenerationTag {};
struct PlanBufferGenerationTag {};
struct ModelIdTag {};
struct ReplicaIdTag {};
struct WorkerIdTag {};
struct RankTag {};
struct DeviceIdTag {};
struct BlockIdTag {};
struct PlanBufferSlotTag {};
struct TenantScopeTag {};
struct TokenIdTag {};
struct AllocationIdTag {};
struct PoolIdTag {};
struct PoolSlotIdTag {};
struct PoolGenerationTag {};
struct FenceSlotIdTag {};
struct FenceGenerationTag {};

// StrongId<Tag, Rep>: explicit construction, value access, comparison,
// hashing, and formatting. No implicit integer conversion and no cross-tag
// conversion — comparisons between different StrongId types do not compile.
template <typename Tag, typename Rep>
class StrongId {
 public:
  explicit constexpr StrongId(Rep value) noexcept : value_(value) {}

  StrongId() = delete;

  [[nodiscard]] constexpr Rep value() const noexcept { return value_; }

  friend constexpr bool operator==(const StrongId&, const StrongId&) = default;
  friend constexpr auto operator<=>(const StrongId&, const StrongId&) = default;

  template <typename H>
  friend H AbslHashValue(H hash, const StrongId& id) {
    return H::combine(std::move(hash), id.value_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const StrongId& id) {
    sink.Append(std::to_string(id.value_));
  }

  friend std::ostream& operator<<(std::ostream& os, const StrongId& id) { return os << id.value_; }

 private:
  Rep value_;
};

using RequestId = StrongId<RequestIdTag, uint64_t>;
using SequenceId = StrongId<SequenceIdTag, uint64_t>;
using StepId = StrongId<StepIdTag, uint64_t>;
using ExecutionTicketId = StrongId<ExecutionTicketIdTag, uint64_t>;
using ReservationId = StrongId<ReservationIdTag, uint64_t>;

using RequestEpoch = StrongId<RequestEpochTag, uint32_t>;
using BlockGeneration = StrongId<BlockGenerationTag, uint32_t>;
using PlanBufferGeneration = StrongId<PlanBufferGenerationTag, uint32_t>;

using ModelId = StrongId<ModelIdTag, uint32_t>;
using ReplicaId = StrongId<ReplicaIdTag, uint32_t>;
using WorkerId = StrongId<WorkerIdTag, uint32_t>;
using Rank = StrongId<RankTag, uint32_t>;
using DeviceId = StrongId<DeviceIdTag, uint32_t>;
using BlockId = StrongId<BlockIdTag, uint32_t>;
using PlanBufferSlot = StrongId<PlanBufferSlotTag, uint32_t>;
using TenantScope = StrongId<TenantScopeTag, uint32_t>;

// Token IDs may be negative (for example added/special-token ranges); no
// non-negative assumption is baked into core validation.
using TokenId = StrongId<TokenIdTag, int32_t>;
using AllocationId = StrongId<AllocationIdTag, uint64_t>;
using PoolId = StrongId<PoolIdTag, uint64_t>;
using PoolSlotId = StrongId<PoolSlotIdTag, uint32_t>;
using PoolGeneration = StrongId<PoolGenerationTag, uint64_t>;
using FenceSlotId = StrongId<FenceSlotIdTag, uint32_t>;
using FenceGeneration = StrongId<FenceGenerationTag, uint64_t>;

// Coordinator-owned sequential generator. Explicit initial value; returns
// OutOfRange before wraparound instead of reissuing IDs.
template <typename Id>
class IdGenerator {
 public:
  using Rep = decltype(std::declval<Id>().value());

  explicit IdGenerator(Id initial) noexcept : next_(initial.value()) {}

  IdGenerator() = delete;

  // Returns the next id and advances; fails without advancing once the
  // representation would wrap.
  absl::StatusOr<Id> Next(absl::string_view component) {
    constexpr Rep kMax = std::numeric_limits<Rep>::max();
    if (next_ == kMax) {
      return absl::OutOfRangeError(absl::StrCat(component, ": id generator exhausted"));
    }
    return Id(next_++);
  }

 private:
  Rep next_;
};

}  // namespace inferx

#endif  // INFERX_BASE_ID_H_
