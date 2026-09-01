# Qualification report: simdjson

- Manifest entry: `simdjson` — approved, feature `core`, owner `config`
- Pin: `0a851a64cd984e9e1a6cab93b6e773aa3f4dc30d` (release tag **v4.6.5**, as
  mandated by `docs/milestones/m1.md` section 3)
- License: Apache-2.0, `third_party/simdjson/LICENSE`

## 1. Which InferX contract uses it?

Strict JSON parsing for configuration (M1), workload files, and replay input
(M1.7) behind the private adapter `src/config/internal/simdjson_reader.cc`.

## 2. Required now, deferred, experimental, or rejected?

Required (core profile) from M1.3 on: `inferx_config` parses all JSON input
through it.

## 3. Source and transitive dependencies

Single C++ library; no transitive third-party requirements (pthreads only,
and `SIMDJSON_ENABLE_THREADS=OFF` disables even that for our use).

## 4. Toolchain/C++23 compatibility

Builds with GCC 13/Clang 18 on Ubuntu 24.04 with
`BUILD_SHARED_LIBS=OFF, SIMDJSON_INSTALL=ON, SIMDJSON_ENABLE_THREADS=OFF,
SIMDJSON_DISABLE_DEPRECATED_API=ON, SIMDJSON_DEVELOPER_MODE=OFF` (scoped per
M0's dependency helper). Evidence: `dev-gcc`, `dev-clang`, `asan-ubsan`,
`tsan`, `analysis` lanes since M1.3. Runtime dispatch selects the best SIMD
kernel for the host CPU; behavior differences by backend are covered by the
m1.md section 19 risk row (adapter detects duplicates; unsupported backends
would be disabled via the build capability report).

## 5. Runtime behavior caveats

The C++ API signals errors through error codes (no exceptions on our paths);
the adapter still wraps parsing in a defensive try/catch per ADR 0004. No
threads, no global mutable state (parser instances are local), no JIT.

## 6. API stability and namespaces

Namespace `simdjson::` (dom/ondemand); release-tagged; the pin follows a
maintained release compatible with the compiler floor.

## 7. License/notice obligations and security

Apache-2.0 — notice retention if sources/binaries are redistributed. The
parser is security-relevant (untrusted model metadata/manifests): the
malformed/limit corpus in `tests/unit/config/` and the parser limits
(`m1.md` section 8.4) are the first gates; fuzzing is tracked in M1.7.

## 8. Binary/build/startup cost

Static archive builds in seconds; parse throughput is gigabytes/second class
— negligible next to config/workload sizes.

## 9. Upgrade/rollback procedure

Advance gitlink + manifest + this report together; re-run the config unit
suite (canonical byte-equality tests are the sensitive gate), sanitizer
lanes, and the M1.7 replay corpus; regenerate goldens only with a reviewed
semantic diff.

## 10. Disposition and approvals

**Approved** for the private parser adapter per m1.md section 3 and ADR 0011.
Public headers must not expose simdjson types — enforced by header
self-containment probes (the public `config/` headers include no simdjson).
