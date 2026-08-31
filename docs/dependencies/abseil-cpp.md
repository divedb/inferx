# Qualification report: abseil-cpp

- Manifest entry: `abseil-cpp` — approved, feature `core`, owner `base`
- Pin: `2c004366e983c5be8334ac1a9be24f75666742cd2b8d` (gitlink; lock lives in the gitlink)
- License: Apache-2.0, `third_party/abseil-cpp/LICENSE`

## 1. Which InferX contract uses it?

ADR 0002 selects `absl::Status`/`absl::StatusOr` as the cross-module error representation;
later `base` work uses strings, time, synchronization, and flags utilities. In M0 it is
qualified via `inferx_absl_qualification_test` and the installed-dependency consumer
fixture; no InferX public header exposes Abseil yet.

## 2. Required now, deferred, experimental, or rejected?

Required for test/qualification targets now; becomes a public dependency of
`inferx::base` in M1 (ADR 0002 §5).

## 3. Source and transitive dependencies

Single C++ source dependency; no transitive third-party requirements (C++ standard
library only). No nested submodules at the pin.

## 4. Toolchain/C++23 compatibility

Compiles with GCC 13.3 and Clang 18.1 on Ubuntu 24.04 in C++23 host mode
(`ABSL_PROPAGATE_CXX_STD=ON`, tests off). Evidence: `dev-gcc`, `dev-clang`,
`asan-ubsan`, `tsan`, and `cpu-release` presets; `tests/consumer_absl` builds and
consumes the installed package. No CUDA involvement; host-only.

## 5. Runtime behavior caveats

May throw (`absl::StatusOr` bad-optional-access paths are avoided by contract),
requires RTTI in places, spawns no threads on its own for the facilities used, and uses
no JIT/Python/global mutable state in the linked subset. Adapters keep any exception
inside the adapter per ADR 0004.

## 6. API stability and namespaces

Namespace `absl`; upstream follows live-at-head with CMake package versioning. InferX
pins the exact commit, so upgrades are deliberate qualification events.

## 7. License/notice obligations and security

Apache-2.0 requires notice retention (including `LICENSE` in source and installed
layouts) and NOTICE file handling where present. No known unpatched security issues
affecting the facilities used; revisit at each upgrade.

## 8. Binary/build/startup cost

Static linking of the status/strings subset adds on the order of low single-digit MiB of
build time and small archive size (`libinferx_base` remains < 100 KiB while Abseil
objects live in `libabsl_*` archives); exact numbers are recorded in the M0 build
metrics JSON.

## 9. Upgrade/rollback procedure

Advance gitlink + manifest `revision` together, run the qualification workflow
(manifest check, all CPU presets, `dependency`-label tests, consumer fixtures, SBOM
regeneration), and update this report. Rollback = revert the commit (gitlink and
manifest revert together).

## 10. Disposition and approvals

**Approved** for base error/string/synchronization utilities per ADR 0002 (proposed) and
this qualification evidence. Approving change: the M0 implementation series.
