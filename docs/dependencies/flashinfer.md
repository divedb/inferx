# Qualification report: flashinfer

- Manifest entry: `flashinfer` — candidate, feature `kernels` (M4), owner `ops`
- Pin: `44428003ba219c14b5473fefec8f7bfd4b72178e` (uninitialized in the core profile)
- License: Apache-2.0, `third_party/flashinfer/LICENSE`

## 1. Which InferX contract would use it?

Paged/ragged attention kernels and possible sampling/MoE kernels behind `KernelRegistry`
(M4+, plan section 9.2).

## 2. Required now, deferred, experimental, or rejected?

Candidate, deferred to M4. **Not approved for production use without the M4 native C++
spike.**

## 3. Source and transitive dependencies

The repository is primarily Python-facing packaging (language: Python) wrapping JIT'd
C++/CUDA; the transitive closure includes torch extension tooling for its default
workflow. Qualification must enumerate what a native-C++-only consumption actually
requires and pin that subset; no Python/PyTorch runtime may enter InferX.

## 4. Toolchain/C++23 compatibility

Unverified at the pin. The M4 spike must prove a native C++ build against the accepted
CUDA toolkit/SM list with the host language rules of ADR 0020.

## 5. Runtime behavior caveats

Default upstream paths use JIT compilation at runtime and global module state — both
prohibited by plan invariants (§3.7 hot-path determinism, §5.1). Only ahead-of-time
compiled kernel objects qualify.

## 6. API stability and namespaces

Fast-moving research project API (Python-first); native C++ surface changes frequently.
Kernel adapters must isolate it completely.

## 7. License/notice obligations and security

Apache-2.0. Re-check advisories at qualification (rapid-moving CUDA code is a reasonable
place for memory-safety scrutiny; compute-sanitizer gates apply).

## 8. Binary/build/startup cost

Potentially large: many template instantiations and compilation units. M4 measures the
minimal native subset before acceptance.

## 9. Upgrade/rollback procedure

M4 spike: initialize, isolate the native C++ subset, build + compute-sanitizer +
reference-correctness gates; upgrades repeat the spike suite.

## 10. Disposition and approvals

**Candidate** (deferred), owner `ops`; decision gate at M4.
