# ADR 0003: Dependency management and offline builds

- Status: Proposed
- Date: 2026-08-30
- Deciding owner: repository owner (implementation proceeds provisionally)
- Seed: `docs/milestones/m0.md` sections 6.2, 8; `docs/plan.md` section 17

## Context

The repository already pins seven git submodules. Uncontrolled they make CI slow and
non-reproducible, license review impossible, and offline builds (a stated M0 outcome)
unachievable. `FetchContent`/`ExternalProject` would move dependency acquisition into
CMake configure, coupling network access to every build and hiding the exact revision
behind cache state.

## Decision

1. **Source dependencies:** pinned git submodules are the canonical provider
   (`INFERX_DEPENDENCY_PROVIDER=submodule`). The gitlink is the lock;
   `third_party/manifest.json` records intent (owner, license, features, status) and
   `tools/deps/check_manifest.py` fails on any drift between gitlinks,
   `.gitmodules`, and the manifest.
2. **System dependencies:** `find_package` with enforced minimum versions
   (`INFERX_DEPENDENCY_PROVIDER=system`), for packaging/downstream use.
3. **No network in CMake, ever.** No `FetchContent`, no `ExternalProject`, no
   `git` invocation. A required-but-missing submodule fails configuration with its
   path and the exact `python3 tools/deps/bootstrap.py --profile core` remediation.
4. **Profiles:** `tools/deps/bootstrap.py` initializes only the direct dependencies a
   feature needs (`core` = Abseil, GoogleTest, Google Benchmark in M0). Optional
   dependency directories are never inspected unless their feature is enabled, so a
   default configure is unaffected by missing optional submodules.
5. **Dependency options are scoped:** options set for a dependency apply only inside
   its `add_subdirectory` and are restored afterward (see
   `cmake/InferXDependencies.cmake`); no `set(... CACHE ... FORCE)` leaks into a
   parent project.
6. **Upgrades** are separate reviewable changes: advance the gitlink and manifest in
   one commit, run the qualification workflow (manifest check, dependent builds/tests,
   SBOM refresh), and record the result in `docs/dependencies/<name>.md`.

## Alternatives

- **CMake `FetchContent` with pinned URLs:** rejected; violates the offline-build
   requirement and hides revisions from the gitlink lock.
- **Package manager for sources (vcpkg/Conan):** rejected for M0; adds a second lock
   format and toolchain bootstrapping before any inference code exists. The `system`
   provider keeps the downstream door open.
- **Vendored copies (no submodules):** rejected; loses upstream history and upgrade
   diffs, and bloats the repository.

## Consequences

- Fresh checkouts run `bootstrap.py --profile core` once; every later configure/build/
  test runs offline.
- CI never uses `submodules: recursive`; it runs the profile it needs.
- Every dependency needs a manifest entry and qualification report before an enabled
  feature may reference it.
- Adding a dependency is a multi-file change (`.gitmodules`, gitlink, manifest,
  report); the drift checker enforces consistency.

## Validation evidence

- `python3 tools/deps/check_manifest.py` passes against the checked-in gitlinks and
  `.gitmodules` (run in CI `policy` job and as a `failure`-label drift fixture with a
  mutated manifest).
- Expected-failure fixtures: missing core submodule prints the bootstrap remediation;
  unknown provider lists valid values (`tests/failure/`).
- Fresh-clone CI lanes (`gcc`, `clang`, `install-consumer`) run with only the core
  profile initialized.

## Supersession

Superseded by a new ADR if the project adopts a system package workflow as primary
(for example for distribution packaging); the manifest/drift machinery must be ported
to the new lock source in the same change.
