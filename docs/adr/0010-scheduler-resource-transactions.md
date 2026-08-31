# ADR 0010: Deterministic scheduler and resource transactions

- Status: Proposed
- Date: 2026-08-31
- Deciding owner: repository owner (implementation proceeds provisionally)
- Seed: `docs/milestones/m1.md` sections 4, 12, 17.6

## Context

M1 must be a correctness oracle for later asynchronous backends. Scheduling
therefore cannot depend on request pointers, hash iteration, unbounded
allocation, or capacity counters that can be partially updated. Plans also
outlive a scheduling call while a fake or physical executor owns them.

## Decision

1. The scheduler consumes arrival-ordered, immutable value snapshots. FCFS
   writes selected indices into caller-owned scratch and retains no view.
2. Admission conservatively reserves one sequence and
   `prompt_tokens + max_output_tokens` KV tokens. `ReservationTxn` has one
   tentative owner; destruction rolls back and `Commit()` is no-fail.
3. A fixed `StepPlanPool` owns sequence storage. A move-only lease carries a
   slot generation and cannot be reused until executor acknowledgement.
4. M1 prefill is the complete prompt in one item; decode is one logical token.
   Batch construction stops at the first budget-blocked selected request and
   does not bypass FCFS head-of-line order.
5. Every plan crosses `StepPlanValidator` before submission. Identity, epoch,
   state, reservation, ranges, model, and aggregate resources must match the
   snapshot exactly.

## Alternatives

- Request pointers in policy/plan objects: rejected because registry mutation
  would invalidate them and obscure ownership.
- Incremental KV reservation: deferred to M8; it changes policy semantics and
  would weaken the M1 accounting oracle.
- Heap-allocating a plan per step: rejected because planning has a zero
  measured-region allocation gate and lifetime is easier to prove with leases.
- Skipping an oversized FCFS head: rejected; M8 may introduce an explicit
  fairness policy, but M1 order must remain unambiguous.

## Consequences

Plan slots impose a deliberate in-flight bound. More concurrency is selected
by `plan_buffer_slots` without changing the ownership protocol. Snapshot and
scratch capacity are allocated before the loop. Scheduler APIs remain CPU-only
and expose no simulator or CUDA types.

## Validation evidence

- `inferx_m1_scheduler_test` covers commit/rollback/release, admission classes,
  FCFS ordering, pool exhaustion/generation, prefill/decode ranges, budget
  saturation, and validator failures.
- `inferx_scheduler_benchmark` measures 1/32/256/1024 requests in four workload
  shapes with 30 repetitions and p50/p90/p99 aggregates. On the recorded local
  Clang debug run, the worst 1024-request p99 was below 0.15 ms and measured
  allocations were zero (the acceptance threshold is 1 ms).

## Supersession

An ADR changing admission, fairness, plan ownership, or the planning latency
gate supersedes this record and supplies replacement traces/benchmark data.
