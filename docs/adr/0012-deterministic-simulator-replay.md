# ADR 0012: Deterministic simulator and canonical replay

- Status: Proposed
- Date: 2026-08-31
- Deciding owner: repository owner (implementation proceeds provisionally)
- Seed: `docs/milestones/m1.md` sections 4, 13-16

## Context

Before a GPU backend exists, request lifecycle, cancellation races, resource
ownership, plan lifetimes, and stale completions need an executable oracle.
Wall time, threads, unordered iteration, compiler-specific status text, and
implicit randomness would make failures difficult to reproduce.

## Decision

1. `EngineSimulator` is a calling-thread-only discrete-event loop driven by
   `ManualClock`; it never sleeps, reads wall time, or starts a thread.
2. Equal timestamps use the fixed order: shutdown/cancel/deadline,
   failure/completion, submit/preempt/requeue, then one planning tick.
3. `FakeExecutor` uses checked configured latency and the specified SplitMix
   request/position token mapping. It retains each plan lease through delivered
   completion acknowledgement.
4. Workload and replay are strict schema-v1 UTF-8 JSON Lines. The writer emits
   fixed key order and stable code/reason values; replay reconstructs inputs,
   reruns, and requires byte equality.
5. Invariants run after every timestamp and at finalization. Final ownership
   must be zero even on cancellation, injected failure, or shutdown cleanup.

## Alternatives

- Real-time timers/background workers: rejected because race scheduling and
  timing noise defeat deterministic replay.
- General JSON DOM/writer dependency: rejected; simdjson is confined to strict
  input adapters and the small canonical writer owns output shape.
- Semantic-only replay comparison: rejected because unstable fields/order
  could enter traces undetected.
- Random fake tokens: rejected; request ID plus output position is the complete
  deterministic sampling input in M1.

## Consequences

The simulator validates control-plane correctness, not GPU throughput or model
quality. Trace schema changes require compatibility review. Large stress traces
remain CI artifacts under `out/`; small reviewed goldens are committed.

## Validation evidence

- Unit/component tests cover latency, token generation, fake ticket lifetime,
  failure rules, and workload parsing.
- Integration tests cover FCFS/capacity, cancel/deadline timing, preemption,
  executor failure, shutdown, terminal responses, and zero final ownership.
- Correctness tests compare the reviewed golden, reordered config inputs,
  repeated traces, and serial/batched fake token identities.
- `inferx_simulator_stress --operations=50000` is the presubmit property gate;
  the same binary accepts the required 1,000,000-operation nightly invocation.

## Supersession

An ADR changing event order, fake execution semantics, replay schema, or
determinism requirements supersedes this record.
