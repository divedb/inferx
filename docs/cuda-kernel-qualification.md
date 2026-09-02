# M4 CUDA kernel qualification

The production CUDA baseline is CUDA 13, SM 89, cuBLASLt dense GEMM/logits, and InferX-owned glue
plus contiguous causal attention. CUDA is opt-in and never silently downgraded. The CPU build does
not discover or load CUDA, FlashInfer, or CUTLASS.

## Dependency disposition

| Component | Disposition | Production boundary |
|---|---|---|
| CUDA/cuBLASLt | accepted system dependency under ADR 0020/0028 | CUDA-only targets; explicit row-major descriptors and bounded workspace |
| FlashInfer `44428003...` | rejected by ADR 0029 | uninitialized closure cannot be audited; option `ON` is a configure error |
| CUTLASS `dc45f979...` | deferred and unregistered | uninitialized audit input; option `ON` is a configure error until evidence identifies a gap |
| owned kernels | accepted source boundary under ADR 0030 | private closed dtype shim; explicit stream; no allocation/sync/JIT |

## Registered CUDA envelope

- cuBLASLt GEMM: FP32/FP16/BF16 A and B of one storage family, FP32 accumulation, explicit output
  dtype, row-major `[M,K] * [N,K]^T`, positive K/N, zero-row no-op, SM-specific registration, and
  caller workspace bounded by the key.
- logits: the same prepared adapter with a distinct operation/capability and FP32 output.
- embedding, RMSNorm, half-split RoPE, SwiGLU, and residual: contiguous FP32/FP16/BF16 storage,
  checked grid arithmetic, exact operation-specific alias capability, and zero-token no-op.
- attention: FlashInfer paged decode/prefill mapped from the contiguous
  BSHD cache (16-bit dtypes, head dims 64/128; ADR 0031/0032), plus the
  remaining operator set in the table below.

All host metadata is validated before launch. Once a launch may have mutated
output, no alternate provider is attempted. Immediate errors use the kernels
layer's status classifier; completion is the caller's fence concern.

## Evidence status (2026-09-01)

Available locally:

- CPU-only GCC C++23 build and 16 M4 unit tests pass;
- 75 public headers pass independent self-containment probes;
- CUDA host adapter sources pass strict GCC C++23 syntax checking against local CUDA headers;
- the owned CUDA kernel translation unit compiles ahead of time for SM 89 in a diagnostic NVCC
  pass; and
- the normal CUDA-enabled configure correctly rejects local NVCC 12.0.140 because CUDA 13 is the
  production floor.

Not available on this machine, and therefore not claimed complete: CUDA 13 linking/real launches,
FP32/FP16/BF16 CPU-oracle sweeps, canary and two-stream tests, memcheck/racecheck/initcheck/synccheck,
100,000-cycle stress, direct cuBLASLt overhead, and retained performance JSON. Those are mandatory
supported-GPU CI evidence before a model using these capabilities is marked ready for production.
The CUDA CI lane invokes `tools/ci/run_compute_sanitizer.sh`; it runs the platform GPU suite under
all four tools and writes a machine-readable manifest, but only retained output from the owned CUDA
13 runner counts as qualification evidence.
