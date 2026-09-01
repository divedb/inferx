// Reference bounded channel (m1.md section 7.5; ADR 0013).
//
// Fixed-capacity ring allocated at construction, one mutex, two condition
// variables. Multiple producers/consumer safe (initial engine use is
// MPSC/SPSC). Never grows; no lock-free claims. Race precedence is contract:
//   push observes: closed -> stop_requested -> free capacity -> timeout
//   pop  observes: stop_requested -> queued data -> drained closed -> timeout
// Consequently a stop never transfers a value, close never accepts a new
// value, and an expired deadline is reported only when no higher-priority
// result is observable under the lock. `value` is moved from only after a
// slot is secured; it stays untouched on every non-success result.

#ifndef INFERX_BASE_BOUNDED_CHANNEL_H_
#define INFERX_BASE_BOUNDED_CHANNEL_H_

#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stop_token>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "inferx/base/clock.h"
#include "inferx/base/token.h"

namespace inferx {

enum class ChannelResult : uint8_t {
  kSuccess,
  kFull,
  kEmpty,
  kClosed,
  kTimeout,
  kStopped,
};

template <typename T>
class BoundedChannel {
 public:
  explicit BoundedChannel(QueueCapacity capacity) : ring_(static_cast<size_t>(capacity.value())) {
    // A zero-capacity channel can never transfer a value; reject at
    // construction rather than deadlock at first use.
    if (ring_.empty()) {
      // QueueCapacity is validated upstream (config); this is a programmer
      // error guard, not a runtime path.
      std::abort();
    }
  }

  BoundedChannel(const BoundedChannel&) = delete;
  BoundedChannel& operator=(const BoundedChannel&) = delete;

  ChannelResult TryPush(T&& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return ChannelResult::kClosed;
    }
    if (count_ == ring_.size()) {
      return ChannelResult::kFull;
    }
    PushLocked(std::move(value));
    return ChannelResult::kSuccess;
  }

  // Blocking push with absolute deadline and cooperative stop. The value is
  // moved only on kSuccess.
  ChannelResult Push(T&& value, Deadline deadline, std::stop_token stop) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::stop_callback stop_wakeup(stop, [this] { not_full_.notify_all(); });
    const bool progressed = not_full_.wait_until(
        lock, deadline, [&] { return closed_ || stop.stop_requested() || HasSpace(); });
    // Precedence: closed > stopped > capacity > timeout.
    if (closed_) {
      return ChannelResult::kClosed;
    }
    if (stop.stop_requested()) {
      return ChannelResult::kStopped;
    }
    if (progressed && HasSpace()) {
      PushLocked(std::move(value));
      return ChannelResult::kSuccess;
    }
    return ChannelResult::kTimeout;
  }

  ChannelResult TryPop(T* output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count_ == 0) {
      return closed_ ? ChannelResult::kClosed : ChannelResult::kEmpty;
    }
    PopLocked(output);
    return ChannelResult::kSuccess;
  }

  // Blocking pop with absolute deadline and cooperative stop. Precedence:
  // stop > queued data (including data left after close) > drained closed >
  // timeout.
  ChannelResult Pop(T* output, Deadline deadline, std::stop_token stop) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::stop_callback stop_wakeup(stop, [this] { not_empty_.notify_all(); });
    // The predicate includes closed_ so a close wakes drain-waiting pops
    // immediately instead of re-blocking until the deadline; precedence is
    // still applied under the lock after the wake.
    const bool progressed = not_empty_.wait_until(
        lock, deadline, [&] { return stop.stop_requested() || count_ > 0 || closed_; });
    if (stop.stop_requested()) {
      return ChannelResult::kStopped;
    }
    if (progressed && count_ > 0) {
      PopLocked(output);
      return ChannelResult::kSuccess;
    }
    if (closed_) {
      return ChannelResult::kClosed;
    }
    return ChannelResult::kTimeout;
  }

  // Idempotent; wakes all waiters, rejects future pushes, and lets queued
  // values drain (pop returns kClosed only once empty).
  void Close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  [[nodiscard]] size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
  }

  [[nodiscard]] size_t capacity() const { return ring_.size(); }

  // Destruction requires all users to have stopped: the owning component
  // closes and joins its threads first (ADR 0013).

 private:
  bool HasSpace() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) { return count_ < ring_.size(); }

  void PushLocked(T&& value) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    ring_[tail_] = std::move(value);
    tail_ = (tail_ + 1) % ring_.size();
    ++count_;
    not_empty_.notify_one();
  }

  void PopLocked(T* output) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    // count_ > 0 is the caller-checked ring invariant; head_ always holds a
    // value when count_ > 0.
    *output = std::move(*ring_[head_]);  // NOLINT(bugprone-unchecked-optional-access)
    ring_[head_].reset();
    head_ = (head_ + 1) % ring_.size();
    --count_;
    not_full_.notify_one();
  }

  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::vector<std::optional<T>> ring_ ABSL_GUARDED_BY(mutex_);
  size_t head_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t tail_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t count_ ABSL_GUARDED_BY(mutex_) = 0;
  bool closed_ ABSL_GUARDED_BY(mutex_) = false;
};

}  // namespace inferx

#endif  // INFERX_BASE_BOUNDED_CHANNEL_H_
