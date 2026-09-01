# ADR 0031: kernels backend architecture and provider chain

- Status: Accepted
- Date: 2026-09-01
- Owner: kernels layer

## Context

The M2–M4 kernel implementation is flat and single-backend: seven owned
kernels live in one translation unit under `kernels/cuda/ops/`, the launch
glue lives in `platform/cuda/src/ops/`, and every operator is executed by
InferX-maintained CUDA code. Maintaining performant kernels for every
operator family is a large, ongoing cost that the project should not pay when
qualified high-performance kernel libraries exist. The architecture also
hard-codes CUDA as the only device backend; AMD and NPU backends are expected
later and must not require upper-layer changes.

This ADR introduces the new architecture on the `kernels-architecture`
branch. The existing M2–M4 implementation (flat kernels, `platform/cuda`
ops glue) remains intact and untouched during the transition; retirement is a
separate future decision once the new surface covers every operator.

## Decision

### Directory taxonomy

```
kernels/
├── include/inferx/kernels/     # unified dispatch API (strictly backend-free)
│   └── ops/<category>.h
├── src/                        # neutral dispatcher (builds on CPU presets)
└── cuda/
    ├── dispatch/               # CUDA backend + providers (host code)
    ├── include/                # internal per-category launch shims
    └── ops/<category>/         # 16 categories (ADR taxonomy below)
```

Categories: activation, attention, attn_res, communication, conv, embedding,
gemm, hyperconnection, kvcache, layernorm, metadata, mhc, moe, quantization,
sampling, transform. Categories without operator contracts today are
README-only scaffolds so the taxonomy is stable while contracts land.

Additional backends (`kernels/amd`, `kernels/npu`) are NOT pre-created as
empty directories: a backend is added by implementing `KernelBackend` for a
new `DeviceKind`, registering it, and adding `kernels/<backend>/ops/*`. No
upper layer changes because dispatch resolves by `DeviceKind` at run time.

### Unified interface

`inferx::kernels::Launch<Op>(ops request, KernelExecutionContext, forced)`
resolves in two steps: `KernelExecutionContext::device.kind` selects the
registered `KernelBackend` (explicit `RegisterKernelBackend`, no static-init
magic), then the backend walks a fixed provider preference chain per
operator. `KernelExecutionContext` carries an opaque `stream_handle`, the
compute capability, and an optional neutral `KernelFailureObserver` so
sticky-fault poisoning stays above the kernels layer.

Everything under `kernels/include` must stay free of backend types; a CPU
test enforces this, and CPU presets build the dispatch surface with zero
CUDA discovery.

### Provider preference chain

Per operator, the strongest-first chain is:

1. **hpc-ops** (`ProviderId::kHpcOps`) — Tencent hpc-ops AOT modules.
2. **flashinfer** (`kFlashInfer`) — ahead-of-time instantiations of the
   pinned FlashInfer device-kernel templates; no JIT modules, no cubin
   loading, no Python in the closure.
3. **cutlass** (`kCutlass`) — custom CUTLASS implementations InferX
   maintains for operators the tiers above do not cover.
4. **vendor/owned tail** — cuBLASLt (`kCublasLt`, reserved) and minimal
   InferX-owned kernels (`kInferxOwned`) as the documented last resort.

Availability is a function of the request (dtype/shape envelope) and compute
capability: providers implement `Available(request*, cc)`, a null probe is a
chain-level query. A forced provider that is unavailable is an
`Unimplemented` error, never a silent fallback. Providers that would be
selected merely because they are registered first is not possible: chain
order is fixed and registration order is irrelevant.

### Layering rules

- `kernels/include` is backend-free (policy-tested).
- `.cu` translation units under `kernels/cuda/ops` carry no Abseil types
  (ADR 0030 discipline, extended to the new tree).
- Request validation is shared per operator (`kernels/cuda/dispatch/
  op_validation.*`) instead of duplicated per provider file.
- Buffer-address resolution reuses the header-only platform accessor
  (`inferx::cuda::BufferAccess`) during the transition; a neutral accessor
  replaces it when the M4 platform layer retires.

## Alternatives

- Reorganizing the existing kernels in place was rejected: the branch must
  leave the current implementation runnable while the new surface qualifies.
- A compile-time provider table per operator was rejected: provider choice
  depends on run-time compute capability and request shape.

## Consequences

- Operator coverage on a given architecture equals the union of provider
  availability probes; gaps are explicit `Unimplemented` errors with the
  recorded reason rather than silent behavior differences.
- The FlashInfer/CUTLASS/hpc-ops dependency qualifications, per-operator
  coverage, and remaining gaps are recorded in ADR 0032.
