# Qualification report: benchmark (Google Benchmark)

- Manifest entry: `benchmark` — approved (build/test-only), feature `core`, owner `benchmarks`
- Pin: `192ef10025eb2c4cdd392bc502f0c852196baa48` (release tag `v1.9.5`)
- License: Apache-2.0, `third_party/benchmark/LICENSE`

## 1. Which InferX contract uses it?

`inferx_base_benchmark` proves benchmark registration and machine-readable JSON output
(`--benchmark_format=json`) consumed by CI evidence jobs. Not linked into any installed
library.

## 2. Required now, deferred, experimental, or rejected?

Required when `INFERX_BUILD_BENCHMARKS=ON` (default top-level). Build/test-only.

## 3. Source and transitive dependencies

None beyond the C++ standard library (its optional GoogleTest-based tests are disabled:
`BENCHMARK_ENABLE_TESTING=OFF`, `BENCHMARK_ENABLE_GTEST_TESTS=OFF`).

## 4. Toolchain/C++23 compatibility

Builds with GCC 13/Clang 18 in C++23 mode on the `cpu-release` lane with install and
upstream tests off. Upstream tests run only during upgrade qualification.

## 5. Runtime behavior caveats

Benchmark binaries spawn worker threads by default (controlled registration in M0's
single-case smoke) and write JSON reports; both are test-harness behavior.

## 6. API stability and namespaces

`benchmark::` namespace, release-tagged.

## 7. License/notice obligations and security

Apache-2.0 notice retention applies only if Benchmark sources/binaries are redistributed;
test-only usage in InferX artifacts triggers none. No known security issues.

## 8. Binary/build/startup cost

Small static archive, seconds of build time; only registered when the benchmarks option
is on.

## 9. Upgrade/rollback procedure

Advance gitlink + manifest together, rebuild `cpu-release`, run the benchmark target
with JSON output and validate the artifact parses, regenerate SBOM. Rollback = revert.

## 10. Disposition and approvals

**Approved** as a build/test-only dependency for M0.
