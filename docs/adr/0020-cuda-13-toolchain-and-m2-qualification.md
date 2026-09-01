# ADR 0020: CUDA 13.0 toolchain and M2 GPU qualification

- Status: Accepted
- Date: 2026-09-01
- Deciding owner: repository owner
- Supersedes: [ADR 0005](0005-toolchain-and-cuda-language.md) and
  [ADR 0019](0019-m2-gpu-qualification.md)

## Context

ADR 0005 proposed CUDA 12.8 as a candidate floor, and ADR 0019 initially made
that candidate the required M2 qualification lane. The owned WSL2 GPU runner
instead has CUDA 13.0.88 and an NVIDIA GeForce RTX 4080 SUPER. It has produced
complete real-device M2 test, failure-containment, Compute Sanitizer, stress,
and paired benchmark evidence. No qualifying CUDA 12.8 result exists.

The repository owner selected CUDA 13.0 as the M2 baseline. This record keeps
the established host-language and qualification rules while making the tested
toolkit the enforceable minimum.

## Decision

1. InferX keeps the Ubuntu 24.04 host floors from ADR 0005: GCC 13 or Clang 18,
   CMake 3.28, Ninja 1.11, Python 3.12 for tooling, and C++23 host targets.
2. M2 requires CUDA toolkit, NVCC, and runtime 13.0 or newer, shared cudart,
   and an explicit accepted architecture list. The accepted M2 combination is
   CUDA 13.0.88, GCC 13.3, and SM 89. Older CUDA toolkits fail configuration;
   an older runtime fails capability validation before context construction.
3. CUDA translation units remain isolated at C++20 when the qualified CMake/
   NVCC combination does not expose `cuda_std_23`; host targets remain C++23.
4. The required GPU lane uses a real device, fails on missing/unsupported
   hardware or skipped tests, and independently selects every M2 unit,
   integration, correctness, failure, and stress category.
5. Compute Sanitizer memcheck, racecheck, initcheck, and synccheck are separate
   mandatory invocations. Wrapper/direct microbenchmarks use at least 30 paired
   repetitions and require median host-submission overhead no greater than 3%.
6. Qualification artifacts record commit and dirty state, runner, GPU/SM,
   driver/runtime/toolkit, compiler/NVCC, architecture flags, configuration,
   seeds, commands, exit statuses, counters, and hashes. Canonical records omit
   UUIDs, pointers, hostname, and wall time.
7. The CUDA developer container is pinned to the multi-platform index digest
   for `nvidia/cuda:13.0.0-devel-ubuntu24.04`.

## Alternatives

- Retaining CUDA 12.8 was rejected because it had no real-device qualification
  evidence on the owned runner and would leave M2 blocked on an unproven lane.
- Allowing any CUDA 12.x release was rejected because it would make the tested
  runtime behavior and supported API floor ambiguous.
- Tracking the newest CUDA 13.x image was rejected because a moving toolkit
  would make CI and developer environments irreproducible.
- Compile-only or skipped GPU validation remains rejected because it cannot
  prove asynchronous ownership, sticky-fault containment, or sanitizer safety.

## Consequences

CUDA 12.x is unsupported for M2 and now fails early with actionable guidance.
The owned CUDA 13.0/SM 89 evidence is qualifying rather than supplementary.
Developers need the pinned CUDA 13.0 container or an equivalent 13.0-or-newer
system toolkit. Upgrading the floor again requires a successor ADR, platform
matrix update, pinned container update, and a complete new evidence manifest.

## Validation evidence

On 2026-09-01, WSL2 Ubuntu 24.04 ran CUDA 13.0.88/NVCC 13.0.88 with GCC 13.3
on an NVIDIA GeForce RTX 4080 SUPER (SM 89, 17,170,956,288 bytes; Windows
driver 591.86; CUDA runtime 13000; driver API 13010). The asynchronous device
self-test passed, followed by 45 unit, 10 integration, 2 correctness, 5
isolated failure-containment, and 4 stress tests.

Compute Sanitizer reported zero memcheck errors/leaks, zero racecheck hazards/
warnings, and zero initcheck or synccheck errors. All seven operation-identical
wrapper/direct median comparisons passed the 3% gate after 30 paired
repetitions. The retained `overhead.json` records the exact p50/p90/p99 values
and computed overheads for each run. The clean cross-artifact manifest is at
`out/evidence/m2-cuda13-wsl-final/manifest.json`.

## Supersession

A successor must qualify its exact toolkit/host compiler/SM combination and
update CMake/runtime floors, CI, container digest, supported-platform matrix,
and retained test/sanitizer/benchmark evidence in the same change.
