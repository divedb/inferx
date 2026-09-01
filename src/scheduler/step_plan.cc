#include "inferx/scheduler/step_plan.h"

#include <cassert>
#include <exception>
#include <limits>
#include <new>
#include <utility>

#include "absl/status/status.h"
#include "inferx/base/status.h"

namespace inferx::scheduler {

StepPlanLease::StepPlanLease(StepPlanPool& pool, PlanBufferSlot slot,
                             PlanBufferGeneration generation) noexcept
    : pool_(&pool), slot_(slot), generation_(generation) {}

StepPlanLease::StepPlanLease(StepPlanLease&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)),
      slot_(other.slot_),
      generation_(other.generation_) {}

StepPlanLease& StepPlanLease::operator=(StepPlanLease&& other) noexcept {
  if (this != &other) {
    Release();
    pool_ = std::exchange(other.pool_, nullptr);
    slot_ = other.slot_;
    generation_ = other.generation_;
  }
  return *this;
}

StepPlanLease::~StepPlanLease() noexcept { Release(); }

const StepPlan& StepPlanLease::plan() const {
  if (pool_ == nullptr) {
    std::terminate();
  }
  assert(pool_ != nullptr);
  return pool_->Plan(slot_, generation_);
}

internal::StepPlanBuffer& StepPlanLease::mutable_buffer() {
  if (pool_ == nullptr) {
    std::terminate();
  }
  assert(pool_ != nullptr);
  return pool_->MutableBuffer(slot_, generation_);
}

void StepPlanLease::Release() noexcept {
  if (pool_ != nullptr) {
    pool_->Release(slot_, generation_);
    pool_ = nullptr;
  }
}

absl::StatusOr<std::unique_ptr<StepPlanPool>> StepPlanPool::Create(
    uint32_t slots, SequenceCount max_sequences_per_step) {
  if (slots == 0 || slots > 64 || max_sequences_per_step.value() == 0) {
    return WithErrorReason(
        absl::InvalidArgumentError(
            "scheduler.plan_pool: slots must be 1..64 and sequence capacity positive"),
        ErrorReason::kInvalidConfig);
  }
  try {
    auto pool = std::unique_ptr<StepPlanPool>(new StepPlanPool());
    pool->max_sequences_per_step_ = max_sequences_per_step.value();
    pool->slots_.resize(slots);
    for (Slot& slot : pool->slots_) {
      slot.buffer.sequences.reserve(max_sequences_per_step.value());
    }
    return pool;
  } catch (const std::bad_alloc&) {
    return WithErrorReason(absl::ResourceExhaustedError("scheduler.plan_pool: allocation failed"),
                           ErrorReason::kCapacityExhausted);
  } catch (...) {
    return WithErrorReason(absl::InternalError("scheduler.plan_pool: unexpected exception"),
                           ErrorReason::kInvariantViolation);
  }
}

absl::StatusOr<StepPlanLease> StepPlanPool::Acquire(StepId step, ModelId model,
                                                    MonotonicTime planned_at) {
  bool generation_exhausted = false;
  for (size_t index = 0; index < slots_.size(); ++index) {
    Slot& slot = slots_[index];
    if (slot.leased) {
      continue;
    }
    if (slot.generation == std::numeric_limits<uint32_t>::max()) {
      generation_exhausted = true;
      continue;
    }
    ++slot.generation;
    slot.leased = true;
    slot.buffer.sequences.clear();
    slot.buffer.plan = StepPlan{
        .schema_version = 1,
        .id = step,
        .model = model,
        .planned_at = planned_at,
        .sequences = std::span<const ScheduledSequence>(),
        .resources = StepResourceUse{},
        .buffer_slot = PlanBufferSlot(static_cast<uint32_t>(index)),
        .buffer_generation = PlanBufferGeneration(slot.generation),
    };
    return StepPlanLease(*this, PlanBufferSlot(static_cast<uint32_t>(index)),
                         PlanBufferGeneration(slot.generation));
  }
  if (generation_exhausted) {
    return WithErrorReason(
        absl::OutOfRangeError("scheduler.plan_pool: available slot generation exhausted"),
        ErrorReason::kInvariantViolation);
  }
  return WithErrorReason(absl::ResourceExhaustedError("scheduler.plan_pool: all slots leased"),
                         ErrorReason::kCapacityExhausted);
}

absl::Status StepPlanPool::Validate() const {
  for (size_t index = 0; index < slots_.size(); ++index) {
    const Slot& slot = slots_[index];
    if (slot.buffer.sequences.capacity() < max_sequences_per_step_) {
      return WithErrorReason(absl::InternalError("scheduler.plan_pool: buffer capacity shrank"),
                             ErrorReason::kInvariantViolation);
    }
    if (slot.leased && (slot.generation == 0 ||
                        slot.buffer.plan.buffer_slot.value() != static_cast<uint32_t>(index) ||
                        slot.buffer.plan.buffer_generation.value() != slot.generation ||
                        slot.buffer.plan.sequences.size() != slot.buffer.sequences.size())) {
      return WithErrorReason(absl::InternalError("scheduler.plan_pool: leased slot mismatch"),
                             ErrorReason::kInvariantViolation);
    }
  }
  return absl::OkStatus();
}

size_t StepPlanPool::available() const noexcept { return slots_.size() - leased(); }

size_t StepPlanPool::leased() const noexcept {
  size_t count = 0;
  for (const Slot& slot : slots_) {
    if (slot.leased) {
      ++count;
    }
  }
  return count;
}

internal::StepPlanBuffer& StepPlanPool::MutableBuffer(PlanBufferSlot slot,
                                                      PlanBufferGeneration generation) {
  if (slot.value() >= slots_.size()) {
    std::terminate();
  }
  assert(slot.value() < slots_.size());
  Slot& selected = slots_[slot.value()];
  if (!selected.leased || selected.generation != generation.value()) {
    std::terminate();
  }
  assert(selected.leased && selected.generation == generation.value());
  return selected.buffer;
}

const StepPlan& StepPlanPool::Plan(PlanBufferSlot slot, PlanBufferGeneration generation) const {
  if (slot.value() >= slots_.size()) {
    std::terminate();
  }
  assert(slot.value() < slots_.size());
  const Slot& selected = slots_[slot.value()];
  if (!selected.leased || selected.generation != generation.value()) {
    std::terminate();
  }
  assert(selected.leased && selected.generation == generation.value());
  return selected.buffer.plan;
}

void StepPlanPool::Release(PlanBufferSlot slot, PlanBufferGeneration generation) noexcept {
  if (slot.value() >= slots_.size()) {
    std::terminate();
  }
  assert(slot.value() < slots_.size());
  Slot& selected = slots_[slot.value()];
  if (!selected.leased || selected.generation != generation.value()) {
    std::terminate();
  }
  assert(selected.leased && selected.generation == generation.value());
  selected.leased = false;
  selected.buffer.sequences.clear();
  selected.buffer.plan.sequences = std::span<const ScheduledSequence>();
  selected.buffer.plan.resources = StepResourceUse{};
}

}  // namespace inferx::scheduler
