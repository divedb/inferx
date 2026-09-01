# CUDA runtime contract

CUDA is optional and build-only in M2. CPU presets neither discover nor link
it. Enabling it requires CUDA toolkit 13.0 or newer, shared cudart, and an
explicit accepted `CMAKE_CUDA_ARCHITECTURES` value. ADRs 0016–0018 and 0020 are
normative; real-GPU qualification is mandatory before declaring M2 accepted.

## Device context and health

`DiscoverCudaDevices` returns value-only capabilities and distinguishes driver,
no-device, invalid-device, and unsupported-capability failures.
`ValidateCudaCapabilities` checks SM, UVA, reserve, and budget before context
construction. M2 selects one device per worker-owned `CudaDeviceContext`.

All Runtime API calls go through an explicitly passed immutable `CudaApi`
table so tests can count and inject failures without global replacement.
CUDA headers and handles remain below `platform/cuda`; generic tensor/runtime
headers remain usable in CPU-only builds.

Health is `Healthy -> Poisoned/ShuttingDown -> Closed`. The first poison cause
wins. Sticky asynchronous faults such as illegal address, assert, timeout, and
destroyed context poison the replica; later submissions are rejected. A
`CudaDeviceGuard` restores the previous device explicitly and idempotently.

## Streams, events, and ordering

Compute and transfer streams are nonblocking and never use the default stream.
Disabling the transfer stream aliases copy work to compute explicitly. Fence
events are pre-created with timing disabled and identified by
device/slot/generation. They are one-record-only; unrecorded poll, stale use,
and early reuse fail.

Cross-stream ordering uses event record/wait:

```text
pinned H2D on transfer -> upload event -> compute waits
-> test kernel on compute -> compute-done event -> transfer waits
-> pinned D2H on transfer -> final completion fence
```

Success from copy/launch means queued, not completed. Output becomes readable
and resources reusable only after the final fence is observed and acknowledged.
Normal paths call no host synchronization. The one named device synchronize is
part of bounded shutdown after pending/deferred work has drained.

The test-only `CudaTestPipeline` returns a move-only `CudaTestSubmission` that
owns both staging leases, device-buffer views carved from its fixed workspace
arena, metadata, compute event, and final fence. `Poll()` is its only
pending-state observation;
`FinishCompleted()` acknowledges completion and returns the pinned output.
Destroying a pending submission moves the whole bundle to the device context's
deferred queue, where it remains valid even if the pipeline is
destroyed and is reclaimed only after terminal observation. If submission
state is uncertain after a failed enqueue and failed named recovery sync, the
bundle is retained and the context is poisoned rather than returning any pool
lease early.

## Allocation and copies

`CudaDeviceAllocator` and `CudaPinnedAllocator` reserve tracker capacity before
`cudaMalloc`/`cudaHostAlloc`. Allocation domains retain the API, device,
tracker charge, and health scope. Explicit release occurs only after consumer
completion. A no-throw CUDA abandonment does not call potentially
synchronizing free; it records leaked bytes and poisons the domain.

`CopyAsync` validates ranges, pointer metadata in debug builds, stream/device,
same-device D2D, and exact direction. Normal H2D/D2H accepts pinned host memory
only. Pageable async, managed memory, overlap, and unsupported directions fail
before submission. `MemsetAsync` accepts device memory and a matching stream.

## Metadata, staging, and workspace

`CudaMetadataRing` owns paired pinned/device slots. `Acquire` exposes the host
slot; `SealAndUpload` performs pinned H2D, records the upload dependency, waits
on compute, and exposes an immutable device view. The pair and upload event
remain retained until the consumer fence acknowledgement.

Pinned staging and device workspace use fixed pools. Pipeline input/output
device storage and kernel scratch are checked suballocations of one retained
workspace lease. Oversize work fails rather than falling back to pageable
memory or per-step CUDA allocation. Event, ring, staging, and workspace
geometry is validated against budgets during startup.

## Shutdown

Shutdown is explicit and idempotent. It rejects new work, polls/acknowledges
live and abandoned fences until a deadline, returns `DeadlineExceeded` without
discarding state when pending work remains, performs the named sticky-status
sync, explicitly frees backing allocations, closes events/streams/domains,
validates tracker baseline, and marks the context closed. Destruction is not a
substitute for this sequence.

## Diagnostics and qualification

`inferx-device-info --json` emits stable device, toolkit/runtime/driver,
accepted/actual SM, budget/pool, and health fields without UUID, pointer,
hostname, or wall time. `--self-test` runs the asynchronous round trip and
returns nonzero for no device or unsupported hardware.
`inferx_cuda_failure_child` isolates illegal-address and device-assert fixtures
so a sticky CUDA process state cannot contaminate another test.

The required evidence flow is:

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release --parallel
ctest --preset cuda-release -L 'm2-unit|m2-integration|m2-correctness|m2-failure|m2-stress'
tools/ci/run_compute_sanitizer.sh --preset cuda-release --suite m2 \
  --output out/sanitizer/m2
tools/bench/run_m2_cuda.sh --preset cuda-release --output out/benchmarks/m2
```

A compile-only or no-device run is not qualification evidence.
