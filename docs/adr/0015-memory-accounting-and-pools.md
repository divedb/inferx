# ADR 0015: Memory accounting and fixed pools

- Status: Accepted
- Date: 2026-08-31
- Deciding authority: `docs/milestones/m2.md` sections 4 and 8

## Context

Inference memory is bounded before a request reaches CUDA. Allocation failure,
pool exhaustion, stale release, and teardown must leave a state that can be
explained without relying on allocator internals or pointer values.

## Decision

1. `MemoryTracker` owns immutable limits by `(device, memory kind)` and stable
   counters by category. A move-only reservation is acquired before allocation,
   then committed to one `AllocationCharge` or rolled back without mutation.
   The tensor layer exposes a narrow accounting transaction interface;
   `MemoryTracker` implements it so a supplied tracker can account
   `CpuAllocator` allocations without reversing dependency direction.
2. Successful free drops committed bytes. An unrecoverable CUDA free moves the
   bytes to `leaked`, retains the live-allocation diagnostic, and poisons its
   domain. Snapshots are thread-safe and key ordered.
3. `FixedBufferPool` divides one checked backing allocation into aligned,
   equal-sized slots. Acquisition picks the lowest free slot and increments a
   generation. Wrap retires the slot; stale, duplicate, and foreign tokens
   never affect a current owner.
4. The production pool is worker-owned and has no internal mutex or hot-path
   allocation. `MutexFixedBufferPool` is the deliberate MPMC/TSan adapter and
   serializes acquire and lease release through a shared adapter mutex.
5. Workspace uses a fixed pool plus checked bump allocation. Failed alignment
   or capacity checks leave its offset unchanged.
6. Fake allocators and fences are test-only, deterministic, bounded, and expose
   the same metadata and lifetime failures as production boundaries.

## Alternatives

- Accounting after allocation was rejected because it can exceed a budget
  between allocation and charge.
- Size-only global accounting was rejected because device, memory kind, and
  category are required for diagnosis and policy.
- Growable/free-list pools and per-step CUDA allocation were rejected because
  warmed execution must allocate neither host nodes nor CUDA storage.
- Slot index without generation was rejected because a late release could free
  a new owner.
- Making the production pool MPMC was rejected because CUDA resources have one
  worker owner; concurrency belongs in the explicit test adapter.

## Consequences

Callers must explicitly close pools and reach tracker baseline. Pool geometry
is immutable, exhaustion is normal backpressure, and generations consume a
finite identity space. A future stream-ordered allocator may replace backing
allocation only if it preserves lease, accounting, and generation semantics.

## Validation evidence

`inferx_runtime_test` covers transactions, limits, peak/leak accounting,
fake failures, deterministic slot order, stale release, and workspace rollback.
`inferx_runtime_stress_test` covers 100,000 generation cycles and the MPMC
adapter; the `tsan` preset owns the race check.

## Supersession

A replacement pool/accounting ADR must retain exact failure accounting and
provide a compatibility path for `PoolToken` and `AllocationCharge` lifetimes.
