# ADR 0006: CI lanes and supported platforms

- Status: Proposed
- Date: 2026-08-30
- Deciding owner: repository owner (implementation proceeds provisionally)
- Seed: `docs/milestones/m0.md` sections 6.1, 11

## Context

M0 must define which CI jobs are required before merge versus advisory, who owns the
runners, and how platform claims are classified. GitHub-hosted runners have no NVIDIA
GPUs, so an honest GPU lane requires an explicitly managed runner; treating a skipped
GPU test as green would produce false coverage precisely where correctness matters most.

## Decision

1. **Required presubmit (CPU, GitHub-hosted ubuntu-24.04):** `policy` (manifest,
   format, docs links, no tracked build outputs), `gcc`, `clang`, `asan-ubsan`,
   `tsan`, `analysis` (clang-tidy, IWYU, header self-containment), `install-consumer`,
   `build-metrics`, `sbom`. A pull request does not merge while any of these is red.
2. **GPU lane:** `ci-gpu.yml` runs on an owned/self-hosted GPU runner (seed evidence:
   RTX 4080, sm_89), triggered manually and on protected `main` changes initially.
   It becomes a **required** presubmit before M2 merges. A GPU job treats "no visible
   device", unsupported compute capability, or a skipped `gpu`-label test as failure.
   Local/developer no-GPU runs may skip with CTest code 77 and a printed reason.
3. **Platform classification** (`docs/supported-platforms.md`): `required` (every PR),
   `supported` (regularly tested on an owned runner), `experimental` (may build, does
   not block), `unsupported` (rejected at configure when reliably detectable).
4. **Hygiene:** third-party actions pinned to commit SHAs with release comments;
   minimal `permissions`; concurrency groups cancel superseded PR runs; bounded job
   and test timeouts; only logs/metrics/test-XML/SBOM uploaded; no flaky-retry
   red-to-green.
5. **Caching:** enabled only after an uncached lane proves clean reproducibility; keys
   include OS image, compiler, CMake, preset, and core gitlink revisions.

## Alternatives

- **GPU-as-optional forever:** rejected; M2+ correctness depends on real device runs.
- **Third-party GPU runner services before an approved security model:** rejected;
   untrusted PR code on shared GPUs with exposed devices is unacceptable (plan section
   18.4 tiering instead).
- **One mega-lane:** rejected; failures would be undebuggable and queue times would
  dominate.

## Consequences

- Repository owner must provision (or accept) the owned GPU runner and its isolation
  model before M2; until then GPU evidence is advisory but still recorded.
- CI differences between lanes are visible in `docs/supported-platforms.md`, not
  folklore.
- Adding a required lane is a change to this ADR's table plus the workflow file.

## Validation evidence

- `.github/workflows/ci-cpu.yml` jobs map one-to-one to the required list; the
  `policy` job runs `tools/deps/check_manifest.py` and `format-check`.
- `.github/workflows/ci-gpu.yml` records device/toolchain evidence and runs the
  non-skipped `gpu` CTest label plus `compute-sanitizer` memcheck.
- `tests/failure/` fixtures prove requested-but-missing CUDA fails configuration.

## Supersession

Superseded when the lane set, required/advisory split, or runner ownership changes;
the supported-platforms matrix must be updated in the same change.
