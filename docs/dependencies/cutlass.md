# Qualification report: cutlass

- Manifest entry: `cutlass` — qualified-deferred, feature `kernels` (M4), owner `ops`
- Pin: `dc45f979ae336a235da1676b311f35efeb30149a` (uninitialized in the core profile)
- License: BSD-3-Clause, `third_party/cutlass/LICENSE.txt` (verified at the pin)

## 1. Which InferX contract would use it?

C++ CUDA templates for specialized GEMM/fusions and reference tuning tools behind the
`KernelRegistry` (plan section 9.2); cuBLASLt remains the default dense GEMM backend.

## 2. Required now, deferred, experimental, or rejected?

Deferred to M4. Not initialized by any M0 profile.

## 3. Source and transitive dependencies

Header-only templates plus optional tools with Python/CMake helper scripts (tools only;
never in the runtime). No required nested submodules for the header path used later.

## 4. Toolchain/C++23 compatibility

Unverified at the pin — this is the deferral reason. CUTLASS is sensitive to NVCC
versions; the M4 qualification must record, for the accepted toolkit (ADR 0005):
compilable example set, supported SM list vs. our matrix, host compiler pairing, and the
`.cu` language level actually usable (C++20 isolation may apply per ADR 0005 §4).

## 5. Runtime behavior caveats

Template/device code; no threads, no global mutable state, no JIT in the kernel path
(examples/tools may use Python tooling which stays out of the runtime). Exceptions only
in host-side tooling/asserts.

## 6. API stability and namespaces

`cutlass::` with example-level API churn between releases; pin exact revisions and adapt
in kernel adapters only.

## 7. License/notice obligations and security

BSD-3-Clause (NVIDIA) — notice retention. No known security issues for library use;
re-check at qualification.

## 8. Binary/build/startup cost

Header-only instantiation is compile-time heavy; contain it to `kernels/` TUs and
measure in M4 before accepting.

## 9. Upgrade/rollback procedure

Qualification at M4: initialize `kernels` profile, build the spike targets against the
accepted CUDA pin + SM list, run correctness against reference GEMMs, record numbers;
upgrades re-run the same.

## 10. Disposition and approvals

**Qualified-deferred** to M4. Owner: `ops`.
