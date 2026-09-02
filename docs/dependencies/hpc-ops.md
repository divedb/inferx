# Qualification report: hpc-ops (Tencent)

- Manifest entry: `hpc-ops` — experimental, disabled; feature `experimental-kernels`;
  owner `ops`
- Pin: `f39028d9f5ab77f71906fbf929d1b611859ab6b7` (uninitialized; audit only)
- License: MIT **with third-party component exceptions** (`LICENSE.txt` at the pin —
  bundled components keep their original licenses and must be enumerated before any use)

## 1. Which InferX contract would use it?

Candidate optimized attention/MoE/sampling/fused-communication kernels as an
experimental SM90+ backend/benchmark source (plan section 9.2: M12/M16 horizon).

## 2. Required now, deferred, experimental, or rejected?

**Experimental and disabled.** Not initialized by any profile; no target reads the
directory.

## 3. Source and transitive dependencies

Upstream is Python-packaging-first with CUDA sources and hardware-specific dispatch;
constraints recorded below must be audited before any inclusion. Nested third-party
components exist under its own exceptions clause — enumerate before use.

## 4. Toolchain/C++23 compatibility

Unverified. Its kernels target SM90+ (higher than the owned RTX 4080 / SM89 runner),
and its build/packaging is Python-toolchain oriented. Both must be qualified separately
if ever promoted.

## 5. Runtime behavior caveats

Python-facing integration and packaging in the default workflow are disallowed in
InferX; only ahead-of-time compiled kernel objects behind the `KernelRegistry` could
qualify. JIT/global-state caveats mirror FlashInfer's.

## 6. API stability and namespaces

Research-project stability; treat as unstable and adapter-isolated.

## 7. License/notice obligations and security

MIT core license with explicit third-party exceptions — any future promotion must
inventory those components and their licenses. No advisories reviewed (nothing is
consumed).

## 8. Binary/build/startup cost

Not measured (nothing built). Record if promoted.

## 9. Upgrade/rollback procedure

Promotion (if ever) is its own ADR + qualification: enumerate components, build the
native subset for the accepted SM list, compute-sanitizer + reference correctness.

## 10. Disposition and approvals

**Experimental, disabled.** Owner: `ops`. Retained as a source of candidate kernels and
benchmarks only.

## Update 2026-09-01 (kernels-architecture branch, ADR 0032)
- Status changed to `approved` (flashinfer/cutlass) / `candidate` (hpc-ops) for
  the ADR 0031 provider chain: integration is ahead-of-time compilation of the
  pinned sources only — no JIT kernel cache, no runtime cubin loading, no
  Python in the link closure. FlashInfer integration additionally requires its
  recorded nested CCCL closure (`3rdparty/cccl` at the flashinfer pin).
- Verified on the SM89 runner (CUDA 13.0): FlashInfer RMSNorm/RoPE device
  templates and the CUTLASS 4.8 classic GEMM pass CPU-oracle tests
  (`inferx_kernels_cuda_test`).
- hpc-ops remains probe-only: its build supports SM90/100/103 exclusively and
  its clean-header kernels do not match the current operator contracts
  (FP8-only GEMMs, fused activations, fused rope+norm+KV-store). Attention on
  SM90+ is the primary future integration target.
- Outstanding evidence: compute-sanitizer sweeps, multi-stream races, FP16/BF16
  oracle suites, binary-size accounting (ADR 0032 gap log).
