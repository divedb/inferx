# ADR 0005: Toolchain floor and CUDA language level

- Status: Proposed
- Date: 2026-08-30
- Deciding owner: repository owner (implementation proceeds provisionally)
- Seed: `docs/milestones/m0.md` sections 6.1, 10

## Context

M0 must state a compiler/CMake/CUDA baseline so CI lanes, containers, and the
supported-platform matrix are concrete instead of "whatever configured once". Host code
is C++23 (ADR 0001). NVCC historically lags the host standard; whether `.cu` translation
units may be C++23 or must be isolated at C++20 depends on the qualified toolkit, so the
decision records both the floor and the isolation rule.

## Decision

1. **Host toolchain floor (required lanes):** GCC 13 and Clang 18 on Ubuntu 24.04,
   CMake ≥ 3.28, Ninja ≥ 1.11, Python 3.12 (tooling only). Exact versions and container
   digests are recorded in `docs/supported-platforms.md`.
2. **Host language:** all InferX host targets use `CXX_STANDARD 23`,
   `CXX_STANDARD_REQUIRED ON`, `CXX_EXTENSIONS OFF`, exported publicly via
   `cxx_std_23`.
3. **CUDA toolkit:** 12.8 is the qualification candidate; it is *accepted* only after
   the owned GPU runner proves configure/build/run against the target GPUs. Local M0
   evidence also exists on CUDA 13.0 (see below); the support matrix records both as
   candidate lanes pending owner acceptance.
4. **CUDA language level:** target C++23 for `.cu` translation units when the
   qualified NVCC supports it; if the qualified NVCC requires lower, isolate the CUDA
   subset at C++20 via a per-target standard instead of lowering the host project.
   The CMake CUDA module selects automatically and records the choice in
   `inferx-info --build`.
5. **Explicit architectures:** `CMAKE_CUDA_ARCHITECTURES` must be an explicit accepted
   list (seed: 89 for the owned RTX 4080 runner); `native` is forbidden in release/CI
   presets. Developers override via `CMakeUserPresets.json`.
6. **Unsupported toolchains** fail at configure time with the minimum required version
   when detection is reliable; a one-off configure is never recorded as support.

## Alternatives

- **GCC 14 / Clang 19 floor:** rejected; Ubuntu 24.04 ships GCC 13/Clang 18 as the
  maintained system compilers, keeping developer containers and CI simple.
- **CMake 3.25 floor:** rejected; 3.28 gives mature presets/install/CUDA behavior and
  is the 24.04 apt version.
- **CUDA C++23 unconditionally:** rejected without toolkit evidence; NVCC pins behind
  the host compiler, so the isolation rule is the robust default.
- **`native` architectures in presets:** rejected; CI must be reproducible and not
  depend on which runner picked up the job.

## Consequences

- `cmake/InferXProjectOptions.cmake` enforces the floor; `cmake/InferXSanitizers.cmake`
  rejects CPU sanitizer presets combined with CUDA in M0.
- The CUDA smoke sets its own standard (C++20 today on NVCC 12.x, C++23 where
  supported); M2 replaces the smoke wholesale, so the split is temporary anyway.
- Upgrading the floor (compiler or CUDA) is a qualification workflow change: new lane
  evidence in `docs/supported-platforms.md`, then a superseding ADR.

## Validation evidence

- CPU: `dev-gcc`, `dev-clang`, `asan-ubsan`, `tsan`, `cpu-release` presets configure,
  build, and test with GCC 13.3 / Clang 18.1 / CMake 3.28 / Ninja 1.11.
- C++23 library floor proven by `tests/compile/cxx23_smoke.cc`
  (`std::optional` monadic operations, `std::unreachable`, `std::string::contains`,
  `std::to_underlying`, `std::span`, `std::jthread`/`std::stop_token`, concepts;
  `std::expected` on lanes that expose it — documented unavailable on Clang 18 +
  libstdc++ 13 because libstdc++ gates `<expected>` on `__cpp_concepts >= 202002L`
  and Clang reports `201907L`).
- CUDA: `cuda-release` preset (NVCC 13.0, sm_89, RTX 4080) builds and runs the
  `gpu`-label smoke; NVCC 12.0 lane verified with the C++20 isolation path.
  CUDA 12.8 remains candidate until the owned runner matrix is accepted.

## Supersession

Superseded when the floor or CUDA pin changes; the successor must update
`docs/supported-platforms.md` and the containers in the same change.
