# Dependency policy

How InferX declares, acquires, qualifies, upgrades, and removes dependencies. The
authoritative records are [`third_party/manifest.json`](../third_party/manifest.json)
(intent, licenses, status) and the gitlinks (the lock). Policy decisions live in
[ADR 0003](adr/0003-dependency-management.md); per-dependency evidence lives in
[`docs/dependencies/`](dependencies/).

## Rules

1. **CMake never acquires anything.** No `FetchContent`, no `ExternalProject`, no Git
   in configure. Configure/build/test must work offline after bootstrap
   (`tools/deps/bootstrap.py`).
2. **The gitlink is the lock.** The manifest's `revision` field is a mirror; drift
   between the two fails `tools/deps/check_manifest.py`.
3. **Profiles initialize only what a feature needs.** `core` = Abseil, GoogleTest,
   Google Benchmark. Optional dependency directories are never inspected unless their
   feature is enabled; a default configure succeeds with optional submodules missing.
4. **An enabled feature with a missing dependency fails configure immediately**,
   naming the path and the bootstrap profile that initializes it.
5. **No local patches inside submodules.** A required change goes upstream or through
   an owned fork behind an ADR.
6. **A declared submodule is not an approved dependency.** Every entry carries a
   status (`candidate`, `approved`, `qualified-deferred`, `experimental`, `rejected`)
   and a qualification report answering the ten questions in
   `docs/milestones/m0.md` section 8.3.
7. **Removal is atomic.** `.gitmodules`, the gitlink, the manifest, and documentation
   change together; rejected entries keep a report and, while removed, a
   `"removed": true` manifest entry so re-adding them is a visible decision.
8. **Upgrades are reviewable changes.** Advance gitlink + manifest + report in one
   change, run the qualification workflow (manifest check, dependent builds/tests,
   sanitizer lanes, SBOM refresh). Upstream test suites run during upgrade
   qualification only, not every InferX build.
9. **Warnings and caches stay contained.** Dependency options are set in a scope and
   restored (see `cmake/InferXDependencies.cmake`); InferX warning/sanitizer policy
   never leaks into dependencies and vice versa.

## Lifecycle

```text
proposed in an ADR/milestone
  -> submodule pinned (gitlink + manifest entry, status: candidate)
  -> qualification report completed
  -> status: approved (feature may require it) | qualified-deferred | experimental
  -> upgrades via the qualification workflow
  -> removal: rejected status + atomic removal (or retained-by-ADR)
```

## System dependencies

CUDA toolkit, cuBLASLt, and NCCL are system dependencies found with `find_package`
(`INFERX_DEPENDENCY_PROVIDER=system` mode does the same for source dependencies for
packagers). Minimum versions are enforced at configure time. Python is a developer
tooling dependency only (3.12) and never a runtime requirement.
