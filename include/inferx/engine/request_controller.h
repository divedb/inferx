// RequestController: the only RequestContext mutator.
// Prepare performs every fallible/allocation-bearing step; Commit is
// noexcept and only moves prepared values and updates fixed-capacity state.

#ifndef INFERX_ENGINE_REQUEST_CONTROLLER_H_
#define INFERX_ENGINE_REQUEST_CONTROLLER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "inferx/base/clock.h"
#include "inferx/engine/request_context.h"
#include "inferx/engine/request_event.h"

namespace inferx {

// Opaque outside the lifecycle implementation.
class PreparedTransition {
 public:
  PreparedTransition(PreparedTransition&&) noexcept;
  PreparedTransition& operator=(PreparedTransition&&) noexcept;
  PreparedTransition(const PreparedTransition&) = delete;
  PreparedTransition& operator=(const PreparedTransition&) = delete;
  ~PreparedTransition();

 private:
  friend class RequestController;
  struct Impl;
  explicit PreparedTransition(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

class RequestController {
 public:
  // Pure decision + guard validation + materialization of any terminal
  // record. The context is unchanged on any failure.
  absl::StatusOr<PreparedTransition> Prepare(const RequestContext& request,
                                             const RequestEvent& event) const;

  // Stamps a prepared terminal/pending-terminal record before the no-fail
  // commit. Keeping this operation here preserves controller-only mutation.
  void StampTerminalTime(PreparedTransition& transition, MonotonicTime now) const noexcept;

  // Applies the prepared transition: state, counters, epochs, in-flight
  // bookkeeping, terminal marker. Resource acquisition/release around the
  // commit is the coordinator's job (TransitionEffects).
  void Commit(RequestContext& request, PreparedTransition transition) noexcept;
};

}  // namespace inferx

#endif  // INFERX_ENGINE_REQUEST_CONTROLLER_H_
