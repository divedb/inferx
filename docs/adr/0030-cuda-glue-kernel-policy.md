# ADR 0030: owned CUDA glue kernel policy

- Status: Accepted
- Date: 2026-09-01
- Owner: CUDA platform

## Context

Embedding, RMSNorm, RoPE, SwiGLU, residual addition, KV append, and the bounded attention fallback
are too small or too semantic-specific for cuBLASLt, but broad bespoke kernel ownership would create
an unreviewable optimization surface.

## Decision

InferX owns only single-purpose ahead-of-time kernels for those operations. Each launcher validates
the exact contiguous dtype/shape/device/phase envelope, accepts an explicit non-default stream,
performs checked 64-bit work/grid arithmetic, launches without allocation or synchronization, and
routes immediate errors through M2 health classification. FP16/BF16 reduction and nonlinear math
convert to FP32 and cast once at output.

Capabilities are narrow and versioned. The fallback attention capability is limited to the ADR 0029
envelope, including even head dimensions and 512-token prefill. CUDA code has no Abseil dependency;
the private launch shim uses a closed three-value storage enum. The correctness embedding kernel
checks the complete immutable ID vector before any output write, preventing out-of-bounds reads and
partial output on an invalid ID.

## Alternatives

- A generic tensor-expression/JIT layer was rejected as outside M4 and incompatible with offline
  startup.
- Registering optimistic broad predicates before boundary tests was rejected because unsupported
  shapes must fail before launch.
- Fusing dense projections was deferred because cuBLASLt remains the baseline and fusion changes
  key/numeric semantics.

## Consequences

The simple kernels favor auditability over peak performance. An optimized replacement registers a
separate narrower capability only after CPU-oracle correctness, all four Compute Sanitizer tools,
tail/adversarial tests, and representative benchmarks. The simple fallback remains available for
forced diagnosis until another backend covers its whole required envelope.

## Validation evidence

The owned CUDA translation unit compiles ahead of time for SM 89 in a local diagnostic NVCC pass,
and CUDA host launchers pass strict GCC syntax checking. This is compilation evidence only. Required
CUDA 13 real-device correctness, canary, multi-stream, sanitizer, and performance results must be
produced by the supported GPU lane before production readiness.

## Supersession

New owned kernels, fusion, padded layouts, fast math, or retirement of a correctness fallback
requires a new ADR with the complete capability and retained qualification evidence.
