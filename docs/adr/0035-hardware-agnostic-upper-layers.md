# ADR 0035: hardware-agnostic upper layers and the backend boundary

- Status: Accepted
- Date: 2026-09-02
- Owner: runtime/model-execution boundary
- Extends: [ADR 0031](0031-kernels-backend-architecture.md)

## Context

ADR 0031 made the kernels layer backend-abstract: dispatch resolves a
`KernelBackend` by `DeviceKind` at run time, CUDA lives under `kernels/cuda/`,
and AMD/NPU backends are expected later "and must not require upper-layer
changes." The M5 specification (`docs/milestones/m5.md`) and parts of
`docs/plan.md`, however, still name CUDA-specific types at the
model/runtime/execution contract level — `CudaModelInstance`,
`CudaModelLoader`, `CudaExecutionBackend` — and put the CUDA realization in
the removed `platform/cuda` tree. If those contracts were implemented as
written, an AMD (ROCm/HIP) or NPU backend would later need parallel upper-layer
types (`AmdModelInstance`, `NpuModelInstance`) or a breaking rename of every
consumer. No M5 code exists yet, so the contracts can be corrected at zero
code cost.

Implemented reality already satisfies the target shape: nothing under
`src/` or `include/inferx/` outside `kernels/` includes a CUDA header, and the
only hardware-named vocabulary above the kernels layer is `DeviceKind` /
`Device::Cuda()`, the `kCuda*` error reasons, the ops `BackendId` values, and
the `cuda.*` config section — platform *names*, never backend types or APIs.

## Decision

1. **The model, runtime, scheduler, engine, sampling, and operator-contract
   layers are hardware-independent.** They name execution devices only through
   the `DeviceKind`/`Device` vocabulary and never include backend headers or
   types (`cuda*`, HIP/ROCm, NPU SDKs) or call backend APIs. This extends the
   control-plane header ban of `docs/plan.md` section 3 to these layers.
2. **Only the backend layer is hardware-specific.** All hardware-specific
   implementation — kernels, kernel providers, device/stream/event/resource
   management, the execution-backend realization, and error translation —
   lives under `kernels/<backend>/`. The CUDA realization (internally named
   `CudaModelInstance`, `CudaModelLoader`, `CudaWeightArena`,
   `CudaExecutionBackend`, `CudaKernelBackend`, …) lives under `kernels/cuda/`
   (model execution under `kernels/cuda/execution/`).
3. **Upper layers see only neutral contracts**: `ExecutionBackend`
   (`Load(ModelLoadPlan) -> ModelHandle`, `Submit`, `Poll`), the loaded
   `ModelInstance` published through the backend, `ExecutionTicket`, and the
   ops/kernels dispatch surface of ADR 0031. No `Cuda*` (or future `Amd*`,
   `Npu*`) type appears in a header, flow, state machine, build switch, test
   gate, or milestone DoD above the backend boundary.
4. **A new backend is added below the boundary**: a new `DeviceKind` value, a
   `kernels/<backend>/` tree implementing `KernelBackend` and the neutral
   execution contracts, startup registration, and its own enable switch,
   config section, presets, and CI lane. No upper layer changes. This is
   ADR 0031's extension rule applied to the whole stack.
5. **`DeviceKind` stays platform-named (`kHost`, `kCuda` today)** and is the
   single hardware-named vocabulary above kernels/. It is an extensible
   backend-platform enumeration that the kernels registry keys on (one backend
   per kind); `kRocm`/`kAmd`/`kNpu` values arrive with their backends. CUDA
   device-fault error reasons and the `cuda.*` config section follow the same
   rule: platform names in neutral vocabulary, backend semantics behind them.
6. **CUDA-first is a delivery decision, not an architectural coupling.** CUDA
   remains the first and initially only qualified backend; every CUDA-specific
   requirement in a milestone (cuBLASLt, compute-sanitizer, `cudaMalloc`
   counters, the `cuda-release` preset) is qualification evidence for the CUDA
   backend realization, not a property of the neutral contracts.

## Alternatives

- Keep `CudaModelInstance`/`CudaExecutionBackend` in the M5 contracts:
  rejected — the first second-backend would fork the runtime/model layer into
  per-vendor types or force a breaking rename of every consumer.
- A separate top-level `backends/<vendor>/` tree for execution-backends while
  `kernels/` stays operator-only: rejected — two per-vendor trees (e.g.
  `backends/cuda` + `kernels/cuda`) blur ownership of the backend; ADR 0031
  already established one `kernels/<backend>` tree per hardware platform and
  `docs/plan.md` already confines backend SDK includes to `kernels/`.
- Rename `DeviceKind::kCuda` to `kGpu`: rejected — the dispatch registry keys
  on `DeviceKind` with one backend per kind, so the kind must identify the
  platform (CUDA vs ROCm vs NPU), not the device class.
- Generalize now by adding stub `kernels/rocm`/`kernels/npu` trees: rejected —
  ADR 0031 explicitly does not pre-create empty backend trees; a backend lands
  when it is implemented and qualified.

## Consequences

- M5's contracts, target graph, gates, and DoD are written in neutral names;
  its CUDA-specific requirements are scoped to the CUDA backend realization
  and its CI lane (`docs/milestones/m5.md` revised accordingly).
- `docs/plan.md` section 3 gains the layering invariant and section 9's
  "CUDA implementation" section is rescoped as "backend implementations
  (CUDA first)".
- M5's planned-ADR table is renumbered (0036–0041): the numbers it previously
  reserved (0031–0036) were taken by accepted ADRs.
- A CPU policy test mirroring the `kernels/include` backend-free check should
  cover the new neutral execution-contract headers when M5.0 lands (M5.0 gate).
- Neutral contracts may still use cross-backend vocabulary — H2D/D2H, pinned
  host memory, streams/events as concepts — because it maps directly onto
  HIP/NPU equivalents; backend *API and tool names* stay inside backend-scoped
  passages.

## Validation evidence

- `kernels/include` backend-free policy test (existing, ADR 0031) — pattern
  for the runtime/model-contract check added with M5.0.
- Zero CUDA includes outside `kernels/` and `third_party/` today:
  `grep -rn "cuda" include/ src/ --include=*.h --include=*.cc` finds only
  platform-named vocabulary (`DeviceKind::kCuda`, `kCuda*` error reasons,
  `BackendId`, `cuda.*` config field macros), no CUDA types or APIs.
- This refactor itself: `docs/plan.md` and `docs/milestones/m5.md` no longer
  reference `Cuda*` types or `platform/cuda` above the backend boundary.

## Supersession

Superseded by a future ADR if a second backend's qualification forces a
contract change above the kernels boundary (which this decision exists to
prevent); otherwise extended per-backend by the backend's own qualification
ADR.
