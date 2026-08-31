# ADR 0004: Exception, RTTI, visibility, and ABI boundaries

- Status: Proposed
- Date: 2026-08-30
- Deciding owner: repository owner (implementation proceeds provisionally)
- Seed: `docs/milestones/m0.md` section 4; `docs/plan.md` section 5.1

## Context

M0 installs a C++23 static library that later milestones extend with Status APIs,
engine internals, and eventually plugins. Decisions on exceptions, RTTI, symbol
visibility, and ABI stability made casually now would either leak dependency
internals (Abseil throws in places; Boost certainly can) or freeze a premature ABI
that M1–M7 must break.

## Decision

1. **Exceptions:** InferX code does not throw across InferX-owned boundaries.
   Dependencies that throw are caught in their adapter and converted to
   `absl::Status` (ADR 0002). InferX compiles with exceptions **enabled** (host
   toolchains, dependencies, and `std::` facades assume it); the rule is a boundary
   discipline, not a `-fno-exceptions` build.
2. **RTTI:** allowed initially. Host code uses it sparingly; revisiting (for example
   for stripped-down serving binaries) requires evidence of binary-size or security
   benefit and a superseding ADR.
3. **Visibility:** InferX targets use default visibility in M0 and set
   `CXX_VISIBILITY_PRESET hidden` + `VISIBILITY_INLINES_HIDDEN` when the first shared
   library exists; nothing in M0 exports a DLL-interface contract.
4. **Install/export:** M0 installs a **static** `inferx::base` with CMake target
   export; consumers must use the same major toolchain lane (GCC 13 / Clang 18,
   libstdc++ C++23). No stable ABI is promised or tested.
5. **No ABI promise before M7:** the installed interface is for first-party
   consumption and CI validation only. A stable shared-library/plugin ABI is a
   deliberate later decision with its own ADR.

## Alternatives

- **`-fno-exceptions` project-wide:** rejected; Abseil/Boost/test frameworks and the
   standard library assume recoverable exceptions in places; forcing it complicates
   every adapter for no M0 benefit.
- **`-fno-rtti` now:** rejected; no demonstrated M0 benefit, breaks `dynamic_cast`
  tooling, and complicates GoogleTest death tests.
- **Shared library with versioned symbols from day one:** rejected; freezes layout
   before request/tensor/model contracts exist (plan section 1).

## Consequences

- Adapter modules own `try`/`catch` boundaries; reviewers reject raw dependency
  exception paths in headers.
- Install rules ship a static library plus headers and CMake config; the consumer
  test validates exactly that surface.
- M7 (or the plugin milestone) must either accept the toolchain-coupled static
  contract or write the ABI ADR with symbol/versioning policy.

## Validation evidence

- `inferx::base` builds static; `install-consumer` CI job installs and links it from
  an independent CMake project.
- Sanitizer lanes (ASan/UBSan/TSan) run the same code paths, catching boundary
  mistakes that exceptions would mask.

## Supersession

Superseded by the shared-library/ABI ADR (expected around M7) or by an RTTI/exception
revisit ADR; the successor must restate what installed consumers may rely on.
