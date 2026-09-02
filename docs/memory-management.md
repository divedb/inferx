# Memory management, pools, and completion

M2 makes memory capacity and reuse explicit. ADRs 0015, 0017, and 0018 are the
normative lifetime decisions.

## Accounting transactions

`MemoryTracker::Create` receives one immutable limit for each configured
`(Device, MemoryKind)` domain. Categories split that capacity into model
weights, KV cache, workspace, execution metadata, pinned staging, runtime
internal, and test usage.

Allocation follows one transaction:

```text
BeginReservation -> allocate -> Commit -> AllocationCharge
                          \-> failure/destruction -> Rollback
```

The reservation checks total reserved + committed + leaked bytes before an
allocator is called. Successful release removes committed bytes and the live
allocation. A CUDA free that cannot be trusted moves committed bytes to
`leaked` while retaining the live diagnostic; it never pretends the capacity
was recovered. `Snapshot()` is thread-safe and stably ordered. Normal shutdown
requires `ValidateBaseline()` success.

`MemoryTracker` implements the tensor layer's device-neutral
`AllocationAccounting` hook. `CpuAllocator(tracker)` therefore follows the
same reserve/commit/release transaction without making the tensor target
depend on runtime. (The M2 CUDA allocators used these transactions
directly; they were removed with the platform substrate, and a future
runtime backend reintroduces device allocators the same way.)

## Fixed buffer pools

`FixedBufferPool::Create` consumes one backing `Buffer` and immutable
`PoolGeometry`. Slots are aligned and equal-sized. Acquire always selects the
lowest free slot, increments its generation, and returns a move-only
`BufferLease`. Pool exhaustion is `ResourceExhausted/kPoolExhausted`.

Releasing requires the exact pool/slot/generation token. Duplicate, stale, or
foreign release is `FailedPrecondition/kStalePoolLease`; generation wrap
retires a slot. The normal pool is single-owner and contains no mutex or
post-creation freelist allocation. `MutexFixedBufferPool` exists solely for
MPMC tests and serializes acquire/release through an adapter mutex.

Call `Close()` after every lease is returned. Closing with a live lease fails
without releasing backing storage. `ValidateInvariants()` is intended at each
debug/test transition.

## Workspace arenas

`WorkspaceArenaPool` applies fixed-pool ownership to scratch arenas.
`WorkspaceLease::Allocate` performs checked alignment and bump allocation;
failure leaves the offset unchanged. Slices never free individually. Closed
`WorkspaceTag` values record per-purpose high-water usage. The entire lease
remains owned through its consumer fence.

## Completion fences

`CompletionFence` is a move-only, hardware-neutral observation token with a
non-owning domain. Its normal lifecycle is:

```text
Acquire/submit -> Poll pending -> Poll complete/failed -> Acknowledge
```

A terminal state must be observed before acknowledgement. Acknowledgement is
exactly once and authorizes the domain to reuse the completion slot. Destroying
an active fence abandons it into deferred reclaim without waiting. Named waits
are limited to test, startup, diagnostic, and shutdown paths and are deadline
bounded.

A fence does not own buffers by itself. A submission/resource bundle must keep
metadata, staging, the workspace lease backing its device-buffer views, and its
fence together until acknowledgement. Early lease release is a lifetime error
even if submission returned successfully.

## Failure policy

- Exhaustion is backpressure and does not mutate an unrelated slot.
- Checked arithmetic failure never advances a reservation or arena.
- Invariant failures use `Internal/kInvariantViolation`; suspect resources are
  not repaired or made reusable.
- Explicit close/release reports errors. Destructors never hide a wait.
- Tests use `FakeAllocator` and `FakeFenceDomain` for deterministic ordinal,
  threshold, category, completion, failure, abandon, and stale-token cases.
