# ADR 0017: Stream, event, and fence contract

- Status: Accepted
- Date: 2026-08-31
- Deciding authority: `docs/milestones/m2.md` sections 4 and 10

## Context

CUDA submission is asynchronous, while buffers, metadata, and events have
finite reusable storage. A raw event handle does not say whether it was
recorded, who may recycle it, or whether an earlier generation owns it.

## Decision

1. Compute and optional transfer streams are explicitly created with
   `cudaStreamNonBlocking`; the default stream is never used. Device and role
   are carried by the wrapper and checked at cross-stream operations.
2. Fence events are pre-created with `cudaEventDisableTiming`. Timing events
   live in a separate benchmark pool.
3. Event acquisition returns one move-only, one-record-only lease identified
   by device, slot, and generation. An unrecorded event cannot be polled or
   waited on; a pending event cannot be reused; generation wrap retires it.
4. `CompletionFence` contains a non-owning `FenceDomain` and move-only token.
   Poll maps not-ready to pending. A terminal state must be observed before
   exactly one acknowledgement returns the slot.
5. Destruction of an active fence abandons it into deferred reclaim without
   waiting. Resource bundles associated with a fence remain owned until the
   same acknowledgement; a fence alone never authorizes early buffer reuse.
6. Host waits are limited to named test, startup, diagnostic, and shutdown
   reasons and use deadline-bounded polling. Normal submission/poll paths have
   no stream/event/device synchronization.

## Alternatives

- Raw events in caller code were rejected because record/ownership state is
  not encoded.
- Event reuse without generations was rejected because late completion or
  acknowledgement could target a newer operation.
- Timing-enabled fence events were rejected because they add overhead to the
  normal path.
- Destructor wait/synchronize was rejected because destruction must not hide
  blocking or discard an error.
- Returning resources when work is merely submitted was rejected because CUDA
  may still be consuming them.

## Consequences

Callers explicitly poll and acknowledge. Event capacity is part of startup
geometry, exhaustion is visible backpressure, and abandoned work needs a
bounded reclaim path. Any future graph or external-event implementation must
continue to satisfy `FenceDomain` generation and acknowledgement semantics.

## Validation evidence

CPU fake-fence unit tests cover pending, complete, failed, abandoned, stale,
and deferred-reclaim transitions. CUDA unit/integration labels exercise real
record/query/wait/ack ordering on the ADR 0019 lane.

## Supersession

A successor may introduce another completion primitive only if it preserves
terminal observation, exactly-once acknowledgement, deferred ownership, and
the no-hidden-wait rule.
