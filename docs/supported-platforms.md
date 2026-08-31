# Supported platforms and toolchains

Classification (see [ADR 0006](adr/0006-ci-and-supported-platforms.md)):

- **required** — built and tested in every pull request;
- **supported** — regularly tested on an owned runner;
- **experimental** — may build; failures do not block a release;
- **unsupported** — rejected at configure time when detection is reliable.

A lane appears here only with evidence (CI job or recorded owned-runner run), never
because it configured once. This matrix is the seed from M0 local evidence; entries
move to `required`/`supported` as the GPU CI lane produces records.

## Host operating system

| Platform | Classification | Evidence |
|---|---|---|
| Ubuntu 24.04 x86-64 | required | `ci-cpu.yml` all jobs; local M0 validation |

## Compilers (host, C++23, `-std=c++23`, no extensions)

| Toolchain | Minimum | Classification | Evidence |
|---|---|---|---|
| GCC 13 (validated 13.3.0) | 13 | required | `dev-gcc`, `cpu-release` presets; configure-time floor check rejects older |
| Clang 18 (validated 18.1.3, libstdc++ 13) | 18 | required | `dev-clang`, `asan-ubsan`, `tsan`, `analysis` presets |
| GCC 12 (NVCC host only) | 12 | supported (CUDA lane only) | NVCC 12.0 toolchain pairing |
| Clang 17 / GCC 14+ | — | experimental | not covered by M0 lanes |
| Anything older than the floor | — | unsupported | `cmake/InferXProjectOptions.cmake` configure failure |

C++23 library facility note (ADR 0005): `std::expected` is available on the
GCC 13 lane but **not** on Clang 18 + libstdc++ 13 — libstdc++ gates
`<expected>` on `__cpp_concepts >= 202002L` while Clang reports `201907L`.
Both lanes remain C++23; the sentinel covers `std::expected` where exposed and
exercises the accepted alternatives (`std::optional` monadic operations,
`std::unreachable`, `std::string::contains`, `std::to_underlying`) on every
lane. A lane that needs `std::expected` must pair Clang with a newer
libstdc++ (or libc++), recorded here when qualified.

## Build tools

| Tool | Minimum | Notes |
|---|---|---|
| CMake | 3.28 (validated 3.28.3) | `cmake_minimum_required(VERSION 3.28)`; older fails before compiling |
| Ninja | 1.11 (validated 1.11.1) | all presets are Ninja single-config |
| Python | 3.12 (validated 3.12.3) | developer tooling only |
| clang-format | major 18 | pinned major; `tools/check_format.sh` enforces |
| clang-tidy | major 18 | `tools/run_clang_tidy.sh` selects `clang-tidy-18` |
| include-what-you-use | 0.21 | analysis preset |

Formatting note: clang-format 18's `Standard` accepts at most `c++20`/`Latest`
(the `c++23`/`c++2b` spellings need 19+), so `.clang-format` uses `Latest`
(deterministic for the pinned major). Revisit with the next pinned major.

## CUDA

| Component | Version | Classification | Evidence |
|---|---|---|---|
| CUDA toolkit 13.0 + GCC 13 host, arch `89` | local dev lane | supported (owned RTX 4080) | `cuda-release` preset, `gpu` CTest label, compute-sanitizer memcheck |
| CUDA toolkit 12.0.140 + GCC 13 host, arch `89`, `.cu` at C++20 | fallback lane | experimental (isolation path) | built and ran the smoke locally; NVCC 12.0 cannot spell C++23 and CMake 3.28 lacks `cuda_std_23`, so `.cu` is isolated at C++20 per ADR 0005 |
| CUDA toolkit 12.8 (candidate) | — | candidate pending qualification | required by `docs/milestones/m0.md` §6.1; accept with owned-runner matrix evidence |
| Driver ≥ 590 (validated 591.86) | — | supported | owned runner record |
| GPU compute capability | `89` (RTX 4080) accepted list | supported | `gpu` smoke fails on devices outside the declared list |
| No toolkit + `INFERX_ENABLE_CUDA=ON` | — | unsupported | configure failure with toolkit/bootstrap guidance |
| `CMAKE_CUDA_ARCHITECTURES` unset or outside list | — | rejected | configure failure listing the accepted architectures |

CPU presets never detect, include, link, or require CUDA (`INFERX_ENABLE_CUDA=OFF`
default). A CPU sanitizer preset combined with CUDA is rejected at configure time.

## Release channel

Public binary distribution is **blocked** until [ADR 0007](adr/0007-project-licensing.md)
is accepted (see `INTERNAL_USE_ONLY.md`).
