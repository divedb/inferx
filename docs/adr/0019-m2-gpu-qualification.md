# ADR 0019: M2 GPU qualification

- Status: Superseded by [ADR 0020](0020-cuda-13-toolchain-and-m2-qualification.md)
- Date: 2026-08-31
- Deciding authority: `docs/milestones/m2.md` sections 3, 17, and 19

## Context

CUDA code that merely compiles does not prove asynchronous lifetime, sticky
fault handling, sanitizer cleanliness, or performance. Results also cannot be
compared or reproduced unless they identify the exact hardware and toolchain.

## Decision

1. M2's minimum qualification toolkit is CUDA 12.8 with shared cudart and the
   explicit accepted architecture list (currently SM 89). The recorded CUDA
   13.0/SM 89 lane remains additional evidence, not a replacement for 12.8.
2. The GPU CI job runs on a real required device, fails when discovery or the
   self-test fails, and runs every `m2-integration`, `m2-correctness`,
   `m2-failure`, and `m2-stress` test without skip-as-success behavior.
3. Compute Sanitizer memcheck, racecheck, initcheck, and synccheck run as
   separate bounded invocations. Any InferX warning/error is failure; expected
   illegal-access fixtures are isolated from the clean suite.
4. Four microbenchmarks cover copy bandwidth, event/fence submission, kernel
   launch submission, and memory-pool versus raw allocation. After warm-up,
   wrapper/direct comparisons use at least 30 paired repetitions and require
   median overhead no greater than 3%, absent a superseding evidence-backed
   ADR.
5. Retained artifacts identify commit, dirty state, runner, GPU/SM, driver,
   runtime/toolkit, compiler/NVCC, architecture flags, config, seeds, command,
   tool version, exit status, counters, and result JSON/log hashes. UUID,
   pointers, hostname, and wall time are excluded from canonical goldens.
6. Host-only compilation, emulation, mocked cudart, or a no-device result is
   useful preflight evidence but cannot close the M2 GPU gate.

## Alternatives

- Compile-only CUDA validation was rejected because it cannot observe ordering
  or sticky asynchronous faults.
- Optional/skipped GPU CI was rejected because it turns lost hardware access
  into a green result.
- One combined sanitizer command was rejected because tool diagnostics and
  expected outcomes must be attributable and bounded.
- Universal latency thresholds were rejected because hardware and power modes
  differ; only paired wrapper overhead is a gate.
- Evidence without exact environment metadata was rejected as irreproducible.

## Consequences

The implementation may be developed on CPU or another CUDA toolkit, but the
milestone is not qualified until the required runner publishes its manifest,
clean sanitizer logs, correctness/stress results, and benchmark JSON. A runner
outage blocks qualification rather than weakening the gate.

## Validation evidence

`.github/workflows/ci-gpu.yml`, `tools/ci/run_compute_sanitizer.sh`,
`tools/bench/run_m2_cuda.sh`, `inferx-device-info --json --self-test`, and the
M2-labeled CTest suites define the evidence flow.

On 2026-09-01, the host WSL2 `Ubuntu-24.04` distribution ran the additional
CUDA 13.0.88/GCC 13.3/SM 89 lane on an NVIDIA GeForce RTX 4080 SUPER (Windows
driver 591.86, runtime 13000, driver API 13010). The asynchronous self-test and
all real-device M2 suites passed: 45 unit, 10 integration, 2 correctness, 5
isolated failure-containment, and 4 stress tests. Compute Sanitizer reported
zero memcheck errors/leaks, zero racecheck hazards/warnings, and zero initcheck
or synccheck errors. All operation-identical wrapper/direct benchmark medians
were within the 3% gate after 30 paired repetitions; the event comparator
prevents interprocedural specialization that is unavailable to the separately
compiled production library.

This is retained real-device evidence for the additional CUDA 13.0 lane. It is
not a qualifying replacement for the required CUDA 12.8 run, which remains
pending because that toolkit is not installed on the runner.

## Supersession

Changing toolkit floor, accepted SMs, runner class, sanitizer matrix, evidence
schema, or the 3% paired gate requires a new qualification ADR and supported-
platform update.
