// Immutable pooled step-plan schema and move-only lease.

#ifndef INFERX_SCHEDULER_STEP_PLAN_H_
#define INFERX_SCHEDULER_STEP_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/scheduler/work_kind.h"

namespace inferx::scheduler {

struct KvReadPlan {
  SequenceId sequence{0};
  TokenRange logical_tokens{TokenOffset(0), TokenOffset(0)};
};

struct KvWritePlan {
  ReservationId reservation{0};
  TokenRange logical_tokens{TokenOffset(0), TokenOffset(0)};
};

struct ScheduledSequence {
  RequestId request{0};
  SequenceId sequence{0};
  RequestEpoch epoch{0};
  WorkKind kind = WorkKind::kPrefill;
  TokenOffset output_position{0};
  TokenRange input_tokens{TokenOffset(0), TokenOffset(0)};
  KvReadPlan kv_read;
  KvWritePlan kv_write;
};

struct StepResourceUse {
  SequenceCount num_sequences{0};
  TokenCount num_tokens{0};
};

struct StepPlan {
  uint32_t schema_version = 1;
  StepId id{0};
  ModelId model{0};
  MonotonicTime planned_at{};
  std::span<const ScheduledSequence> sequences;
  StepResourceUse resources;
  PlanBufferSlot buffer_slot{0};
  PlanBufferGeneration buffer_generation{0};
};

namespace internal {

struct StepPlanBuffer {
  StepPlan plan;
  std::vector<ScheduledSequence> sequences;
};

}  // namespace internal

class BatchPlanner;
class StepPlanPool;

class StepPlanLease {
 public:
  StepPlanLease(StepPlanLease&& other) noexcept;
  StepPlanLease& operator=(StepPlanLease&& other) noexcept;
  StepPlanLease(const StepPlanLease&) = delete;
  StepPlanLease& operator=(const StepPlanLease&) = delete;
  ~StepPlanLease() noexcept;

  [[nodiscard]] const StepPlan& plan() const;
  [[nodiscard]] bool valid() const noexcept { return pool_ != nullptr; }

 private:
  friend class BatchPlanner;
  friend class StepPlanPool;
  StepPlanLease(StepPlanPool& pool, PlanBufferSlot slot, PlanBufferGeneration generation) noexcept;
  [[nodiscard]] internal::StepPlanBuffer& mutable_buffer();
  void Release() noexcept;

  StepPlanPool* pool_ = nullptr;
  PlanBufferSlot slot_{0};
  PlanBufferGeneration generation_{0};
};

class StepPlanPool {
 public:
  static absl::StatusOr<std::unique_ptr<StepPlanPool>> Create(uint32_t slots,
                                                              SequenceCount max_sequences_per_step);

  [[nodiscard]] absl::StatusOr<StepPlanLease> Acquire(StepId step, ModelId model,
                                                      MonotonicTime planned_at);
  [[nodiscard]] absl::Status Validate() const;
  [[nodiscard]] size_t available() const noexcept;
  [[nodiscard]] size_t leased() const noexcept;

 private:
  friend class StepPlanLease;

  struct Slot {
    internal::StepPlanBuffer buffer;
    uint32_t generation = 0;
    bool leased = false;
  };

  StepPlanPool() = default;
  [[nodiscard]] internal::StepPlanBuffer& MutableBuffer(PlanBufferSlot slot,
                                                        PlanBufferGeneration generation);
  [[nodiscard]] const StepPlan& Plan(PlanBufferSlot slot, PlanBufferGeneration generation) const;
  void Release(PlanBufferSlot slot, PlanBufferGeneration generation) noexcept;

  std::vector<Slot> slots_;
  uint32_t max_sequences_per_step_ = 0;
};

}  // namespace inferx::scheduler

#endif  // INFERX_SCHEDULER_STEP_PLAN_H_
