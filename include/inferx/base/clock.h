// Engine clocks and deadlines (m1.md section 7.4; ADR 0008).
//
// All lifecycle and replay time is nanosecond-resolution monotonic time.
// Wall time never appears in lifecycle or replay decisions.

#ifndef INFERX_BASE_CLOCK_H_
#define INFERX_BASE_CLOCK_H_

#include <chrono>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace inferx {

using Nanoseconds = std::chrono::nanoseconds;
using MonotonicTime = std::chrono::time_point<std::chrono::steady_clock, Nanoseconds>;
// Same type as MonotonicTime; the alias documents "must complete before this
// instant". A deadline expires when Now() >= deadline.
using Deadline = MonotonicTime;

class Clock {
 public:
  virtual ~Clock() = default;
  virtual MonotonicTime Now() const noexcept = 0;
};

// Production adapter: the process steady clock.
class SteadyClock final : public Clock {
 public:
  MonotonicTime Now() const noexcept override {
    return std::chrono::time_point_cast<Nanoseconds>(std::chrono::steady_clock::now());
  }
};

// Simulator/test-only controlled clock. Single-threaded by contract (the
// simulator loop and tests advance it from one thread); Now() is noexcept.
// Time only moves forward; overflow is rejected rather than clamped.
class ManualClock final : public Clock {
 public:
  explicit ManualClock(MonotonicTime initial) noexcept : now_(initial) {}

  ManualClock() = delete;

  MonotonicTime Now() const noexcept override { return now_; }

  absl::StatusOr<MonotonicTime> Advance(Nanoseconds duration) {
    if (duration < Nanoseconds::zero()) {
      return absl::InvalidArgumentError("clock.advance: duration must be nonnegative");
    }
    const MonotonicTime::duration epoch_now = now_.time_since_epoch();
    const MonotonicTime::duration epoch_max = MonotonicTime::max().time_since_epoch();
    if (duration > epoch_max - epoch_now) {
      return absl::OutOfRangeError("clock.advance: time point overflow");
    }
    now_ += duration;
    return now_;
  }

 private:
  MonotonicTime now_;
};

}  // namespace inferx

#endif  // INFERX_BASE_CLOCK_H_
