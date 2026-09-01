# ADR 0028: cuBLASLt algorithm and workspace policy

- Status: Accepted
- Date: 2026-09-01
- Owner: CUDA platform

## Context

Dense Llama projections need a production CUDA backend without transposing M3 `[out,in]` weights or
performing heuristic search and descriptor allocation during a live request.

## Decision

Each CUDA device worker owns one `CublasLtContext`. Preparation validates a complete GEMM/logits
key, creates explicit row-major descriptors for `A[M,K]`, `B[N,K]`, and `D[M,N]`, sets `op(A)=N`
and `op(B)=T`, and uses FP32 scale/compute. FP32 selects pedantic FP32 compute; FP16/BF16 select FP32
compute. Logits capabilities require FP32 output.

Preparation queries at most eight heuristic candidates in vendor order, rejects failed or
over-budget results, runs `cublasLtMatmulAlgoCheck`, and retains the first valid algorithm. Workspace
is caller-owned, bounded by the key, and at least 256-byte aligned when nonempty. Opaque algorithms
are process-local and never serialized. A zero-row plan is a validated no-op and makes no cuBLASLt
call.

Launch rechecks shapes, dtypes, device memory, pointer alignment class, finite scales, addend rules,
disjoint allocation intervals, workspace, stream/device, and context health. It creates no
descriptor, performs no heuristic query/allocation/synchronization, and checks immediate vendor and
CUDA launch status. The caller retains the plan, buffers, workspace, stream, and completion fence.

## Alternatives

- cuBLAS column-major reinterpretation was rejected because it obscures source orientation.
- Online benchmarking was rejected because selection would depend on load and mutate first-request
  latency.
- Durable serialization of `cublasLtMatmulAlgo_t` was rejected because it is opaque and version
  scoped.
- Baseline CUTLASS instantiations were rejected because no measured cuBLASLt capability gap exists.

## Consequences

Startup performs bounded descriptor/heuristic work and may fail readiness if no candidate fits.
Actual aligned representative buffers still require CUDA 13 warm-up correctness before a plan is
ready. A toolkit, driver, SM, layout, dtype, or workspace change invalidates local plans.

## Validation evidence

The adapter's CUDA host translation units pass GCC 13 C++23 syntax checks against the locally
available CUDA headers with the project warning set. The supported CUDA 13 real-launch,
direct-call-overhead, and Compute Sanitizer gates cannot run on the current CUDA 12.0 machine; no
performance or numeric qualification is claimed from syntax validation.

## Supersession

Offline benchmark selection, another math mode, durable documented algorithm attributes, or a
CUTLASS backend requires a new ADR and retained numeric/performance evidence.
