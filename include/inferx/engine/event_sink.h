// Production-neutral response and lifecycle observation boundaries.

#ifndef INFERX_ENGINE_EVENT_SINK_H_
#define INFERX_ENGINE_EVENT_SINK_H_

#include <exception>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/api/response_event.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/engine/request_event.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/scheduler/resource_accountant.h"

namespace inferx {

class ResponseReservation {
 public:
  explicit ResponseReservation(ResponseEvent event) : event_(event) {}
  ResponseReservation(ResponseReservation&&) noexcept = default;
  ResponseReservation& operator=(ResponseReservation&&) noexcept = default;
  ResponseReservation(const ResponseReservation&) = delete;
  ResponseReservation& operator=(const ResponseReservation&) = delete;
  ~ResponseReservation() = default;

  [[nodiscard]] bool valid() const noexcept { return event_.has_value(); }
  [[nodiscard]] ResponseEvent Take() noexcept {
    if (!event_.has_value()) {
      std::terminate();
    }
    ResponseEvent result = event_.value();
    event_.reset();
    return result;
  }

 private:
  std::optional<ResponseEvent> event_;
};

class ResponseSink {
 public:
  virtual ~ResponseSink() = default;
  [[nodiscard]] virtual absl::StatusOr<ResponseReservation> Prepare(ResponseEvent event) = 0;
  virtual void Commit(ResponseReservation reservation) noexcept = 0;
};

struct TransitionObservation {
  MonotonicTime at{};
  RequestId request{0};
  SequenceId sequence{0};
  RequestEpoch epoch{0};
  RequestState from = RequestState::kReceived;
  RequestEventKind event = RequestEventKind::kInputReady;
  RequestState to = RequestState::kReceived;
  absl::StatusCode status = absl::StatusCode::kOk;
  ErrorReason reason = ErrorReason::kNone;
};

enum class ResourceAction : uint8_t {
  kReserve,
  kRelease,
};

struct ResourceObservation {
  MonotonicTime at{};
  ResourceAction action = ResourceAction::kReserve;
  ReservationId reservation{0};
  RequestId request{0};
  scheduler::ResourceCost cost;
  scheduler::ResourceSnapshot snapshot;
};

class LifecycleObserver {
 public:
  virtual ~LifecycleObserver() = default;
  [[nodiscard]] virtual absl::Status ObserveTransition(
      const TransitionObservation& observation) = 0;
  [[nodiscard]] virtual absl::Status ObserveResource(const ResourceObservation& observation) = 0;
};

class NullLifecycleObserver final : public LifecycleObserver {
 public:
  absl::Status ObserveTransition(const TransitionObservation&) override { return absl::OkStatus(); }
  absl::Status ObserveResource(const ResourceObservation&) override { return absl::OkStatus(); }
};

}  // namespace inferx

#endif  // INFERX_ENGINE_EVENT_SINK_H_
