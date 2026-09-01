# ADR 0018: Metadata and workspace lifetime

- Status: Accepted
- Date: 2026-08-31
- Deciding authority: `docs/milestones/m2.md` sections 4, 12, and 14

## Context

Kernel metadata is produced by the host, copied to the device, and consumed
asynchronously. Staging and scratch memory follow the same fence lifetime.
Reusing any of them after upload rather than after consumer completion creates
a cross-stream use-after-reuse bug.

## Decision

1. `CudaMetadataRing` preallocates paired pinned-host/device slots with a fixed
   geometry. Slot state and generation cover host-writable, sealed/uploaded,
   in-flight, and free ownership.
2. Sealing performs pinned H2D on the transfer stream, records an upload event,
   and makes compute wait on that event. The device view becomes immutable and
   host writes are rejected after sealing.
3. The upload event and both metadata buffers remain retained through the
   consumer fence. A metadata lease may release only after that fence has been
   acknowledged.
4. Pinned staging comes from a fixed `FixedBufferPool`; oversize/exhausted
   requests fail rather than falling back to pageable memory.
5. Workspace backing is split into fixed arenas. Each `WorkspaceLease` uses
   checked aligned bump allocation and closed-tag high-water accounting. The
   pipeline carves device input, output, and scratch views from that arena, so
   submission performs no per-step CUDA allocation. The arena remains in the
   submission resource bundle until acknowledgement.
6. The test pipeline's final transfer fence covers H2D, compute, and D2H. It
   does not expose output bytes while pending and releases resources in a
   deterministic order after completion.

## Alternatives

- Pageable asynchronous copies were rejected because CUDA may stage or
  synchronize implicitly; pageable transfer exists only in named benchmarks.
- Per-step CUDA allocations were rejected because allocation/free may
  synchronize and make latency unbounded.
- Releasing metadata after upload was rejected because compute has only begun
  its dependency at that point.
- Independently freeing workspace slices was rejected because one fixed arena
  and fence lifetime are easier to validate.
- One unpaired ring with pointer arithmetic was rejected because pinned and
  device ownership/accounting would become ambiguous.

## Consequences

Pool geometry limits maximum in-flight work and is validated before CUDA
allocation. Submission objects must be move-only resource bundles, not just a
fence. Later operators may add stronger geometry formulas but cannot weaken
the acknowledgement reuse point.

## Validation evidence

CPU workspace tests prove no-mutation failure semantics. CUDA integration,
correctness, failure, and 100,000-cycle stress labels validate the paired-ring
and bundle transitions on the ADR 0019 runner.

## Supersession

A successor must prove the same upload dependency and consumer-fence reuse
point for any arena, graph, or stream-ordered replacement.
