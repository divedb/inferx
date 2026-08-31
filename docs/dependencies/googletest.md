# Qualification report: googletest

- Manifest entry: `googletest` — approved (build/test-only), feature `core`, owner `tests`
- Pin: `063de7e9578f82b369302001269680b4b1553359` (release tag `v1.18.0`)
- License: BSD-3-Clause, `third_party/googletest/LICENSE`

## 1. Which InferX contract uses it?

Unit, integration, and expected-failure tests (`inferx_base_test`,
`inferx_absl_qualification_test`, consumer/failure fixtures). GoogleMock is enabled as
the declared test dependency. It is never linked into any installed InferX library.

## 2. Required now, deferred, experimental, or rejected?

Required when `BUILD_TESTING=ON` (default top-level). Build/test-only: absent from the
install rules and from all shipped binaries.

## 3. Source and transitive dependencies

None beyond the C++ standard library.

## 4. Toolchain/C++23 compatibility

Builds with GCC 13/Clang 18 in C++23 mode on the `dev-gcc`/`dev-clang` lanes with
`INSTALL_GTEST=OFF`. Upstream test suite runs only during upgrade qualification, not
every InferX build.

## 5. Runtime behavior caveats

Test binaries only. GoogleTest registers global state for its framework (acceptable in
test targets), spawns threads only for death/multithread tests InferX opts into.

## 6. API stability and namespaces

`testing::` namespace, release-tagged; the pin follows a maintained release compatible
with the compiler floor.

## 7. License/notice obligations and security

BSD-3-Clause notice retention applies to redistributed source or binaries of GoogleTest
itself; since it is test-only and not shipped, no distribution obligation is triggered
by InferX artifacts. No known security issues relevant to test execution.

## 8. Binary/build/startup cost

Builds in tens of seconds on the CI lane; no runtime cost in shipped artifacts.

## 9. Upgrade/rollback procedure

Advance gitlink + manifest together, build all CPU presets, run the full CTest suite
(labels `unit`, `dependency`, `integration`, `failure`), optionally run upstream's own
tests during the upgrade PR, regenerate SBOM. Rollback = revert the commit.

## 10. Disposition and approvals

**Approved** as a build/test-only dependency for M0.
