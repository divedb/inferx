# ADR 0013: Bounded channel semantics

- Status: Proposed
- Date: 2026-08-31
- Deciding owner: repository owner (implementation proceeds provisionally)
- Seed: `docs/milestones/m1.md` sections 4, 7.5

## Context

The future engine (M7) coordinates threads across bounded queues: HTTP I/O
producers into the coordinator, completion events back out. M1 needs the
reference implementation of those boundary semantics — close/backpressure/
stop ordering — tested independently of the (single-threaded, no-wait)
simulator. Folly was rejected at M0; the queue is small project code.

## Decision

1. `BoundedChannel<T>` is a fixed-capacity ring of `std::optional<T>`
   allocated at construction, guarded by one `std::mutex` and two condition
   variables (not-empty, not-full). Multiple producers/consumers are safe;
   initial engine use is MPSC/SPSC.
2. Results are the closed `ChannelResult` set
   (Success/Full/Empty/Closed/Timeout/Stopped). `value` is moved only after a
   slot is secured and remains untouched otherwise, so callers can retry
   move-only work.
3. Race precedence is contract (checked under the mutex):
   push observes closed → stop → capacity → timeout; pop observes stop →
   queued data → drained-closed → timeout. A stop never transfers a value;
   close never accepts a new value; drain after close is guaranteed.
4. Blocking operations install a stack-scoped `std::stop_callback` that
   notifies the relevant condition variable; predicates (not notification
   counts) decide progress. Notifications happen after publishing ring state.
5. Close is idempotent, wakes all waiters, and pops return `kClosed` only
   after the queue is empty. Destruction requires all users stopped; owners
   close and join first.
6. No lock-free claims and no growth: backpressure is the point. Allocation
   happens only at construction (payload-internal allocations are the
   payload's own).

## Alternatives

- **Folly MPMCQueue / higher-throughput queues:** rejected — M0 removed Folly
  for cost; the engine needs correctness-first reference semantics.
- **Lock-free ring (atomics only):** rejected for M1 — much harder close/stop
  proofs, no measured need; revisit behind the same header with a measured
  M7 ADR.
- **Unbounded queue with condition variables:** rejected — hides backpressure,
  violating the roadmap's bounded-cardinality invariant.
- **`std::stop_token`-free API (separate Stop()):** rejected — cooperative
  cancellation through `jthread` machinery is the plan's threading model
  (plan section 5.1).

## Consequences

- The simulator does not use this channel (it never blocks); the channel is
  the tested future engine boundary.
- TSan stress (label `core-channel`) is a CI requirement from M1 on.
- Semantics changes (precedence, drain, results) are breaking: they alter the
  transition-observable behavior of M7 and require a superseding ADR.

## Validation evidence

- `tests/unit/base/channel_test.cc` (labels `core-channel`): capacity 1/N,
  FIFO, full/empty, close/drain, expired deadline, stop-without-transfer,
  move-only payloads with untouched-on-failure checks, and a
  multi-producer/multi-consumer stress run; the TSan lane runs the label.

## Supersession

Superseded by a channel-semantics ADR (precedence/close changes) or by a
qualified lock-free implementation ADR measured in M7.
