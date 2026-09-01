# ADR 0016: CUDA context ownership and error scope

- Status: Accepted
- Date: 2026-08-31
- Deciding authority: `docs/milestones/m2.md` sections 4, 9, and 13

## Context

CUDA calls can fail synchronously, report earlier asynchronous faults, or leave
a context unsafe. M2 needs deterministic tests around those calls and one
place that decides whether a failure affects an operation or the entire device
replica.

## Decision

1. M2 uses the CUDA Runtime API with shared `cudart`. `CudaApi` is an immutable,
   explicitly passed table of typed M2 function pointers; production maps
   directly to cudart and tests may inject/count calls.
2. CUDA headers and raw handles are confined to `platform/cuda` and CUDA test
   kernels. Generic tensor/runtime APIs carry only `Device`, values, views,
   and hardware-neutral fences.
3. One `CudaDeviceContext` is owned by one worker and one selected device. It
   discovers value-only capability data, validates the accepted SM/UVA/budget
   matrix, and constructs resources transactionally.
4. `CudaHealth` transitions from healthy to poisoned, shutting down, then
   closed. First poison cause wins. Illegal address, assert, timeout, context
   destruction, and other sticky asynchronous faults poison the replica and
   reject subsequent work.
5. Stable `ErrorReason` values 100 through 110 distinguish invalid device,
   OOM, launch rejection, async/device loss, API failure, stale fence/pool,
   exhaustion, pending resources, and unsupported capability.
6. Device switching uses a move-only guard with explicit idempotent restore.
   Normal worker execution pins its device once. Shutdown is explicit and is
   the only production path permitted to perform the named device synchronize.

## Alternatives

- The Driver API was rejected because M2 does not need module/context control
  beyond the Runtime API and would duplicate runtime ownership.
- Globally replacing CUDA symbols in tests was rejected because it is unsafe
  under concurrency; the seam is passed explicitly.
- Raw CUDA handles in generic headers were rejected because they would make
  CPU builds depend on CUDA and blur ownership.
- Treating illegal-address or device-lost errors as an operation-only failure
  was rejected because later calls cannot be assumed safe.
- Destructor synchronization was rejected because it hides unbounded blocking
  and errors.

## Consequences

CPU builds do not discover or link CUDA. CUDA resource creation and shutdown
are more explicit, but every failure has a deterministic scope. Adding a CUDA
API call requires extending the seam, tests, error mapping, and boundary check.
Multi-device execution means independent worker contexts, not shared handles.

## Validation evidence

Host translation units compile warning-clean against CUDA headers, the `.cu`
test kernels compile through the isolated language target, and
`m2_cuda_header_boundary` rejects CUDA includes outside the platform boundary.
Runtime qualification is owned by ADR 0019 and is not satisfied by host-only
compilation.

## Supersession

A successor may change API layer or context topology only with equivalent
failure injection, health scoping, and CPU/CUDA header isolation.
