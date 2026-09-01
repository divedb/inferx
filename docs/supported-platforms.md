# Supported platforms and toolchains

Classification (see [ADR 0006](adr/0006-ci-and-supported-platforms.md)):

- **required** — built and tested in every pull request;
- **supported** — regularly tested on an owned runner;
- **experimental** — may build; failures do not block a release;
- **unsupported** — rejected at configure time when detection is reliable.

A lane appears here only with evidence (CI job or recorded owned-runner run),
never because it configured once. This matrix began with M0 local evidence.
M2 does not promote a CUDA entry until the exact ADR 0020 artifact exists.

## Host operating system

| Platform | Classification | Evidence |
|---|---|---|
| Ubuntu 24.04 x86-64 | required | `ci-cpu.yml` all jobs; local M0 validation |
| Ubuntu 24.04 x86-64 on WSL2 | supported CUDA qualification host | 2026-09-01 real-device CUDA 13.0/SM 89 M2 run through the host WSL launcher |

## Compilers (host, C++23, `-std=c++23`, no extensions)

| Toolchain | Minimum | Classification | Evidence |
|---|---|---|---|
| GCC 13 (validated 13.3.0) | 13 | required | `dev-gcc`, `cpu-release` presets; configure-time floor check rejects older |
| Clang 18 (validated 18.1.3, libstdc++ 13) | 18 | required | `dev-clang`, `asan-ubsan`, `tsan`, `analysis` presets |
| GCC 12 (legacy NVCC host only) | 12 | experimental | CUDA 12.0 compile-only history; below the M2 toolkit floor |
| Clang 17 / GCC 14+ | — | experimental | not covered by M0 lanes |
| Anything older than the floor | — | unsupported | `cmake/InferXProjectOptions.cmake` configure failure |

C++23 library facility note (ADR 0020): `std::expected` is available on the
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
| OpenSSL development package | 3.0 | required by the default native Hugging Face HTTPS transport; not required with `INFERX_ENABLE_HF_HUB=OFF` |

Formatting note: clang-format 18's `Standard` accepts at most `c++20`/`Latest`
(the `c++23`/`c++2b` spellings need 19+), so `.clang-format` uses `Latest`
(deterministic for the pinned major). Revisit with the next pinned major.

## CUDA

| Component | Version | Classification | Evidence |
|---|---|---|---|
| CUDA toolkit 13.0.88 + GCC 13.3 host, arch `89` | M2 minimum | required and qualified by ADR 0020 | 2026-09-01 WSL2 run: device self-test; 45 unit, 10 integration, 2 correctness, 5 failure, and 4 stress tests; four clean Compute Sanitizer tools; seven paired benchmark medians within 3% |
| CUDA toolkit 12.x | below M2 minimum | unsupported | CMake rejects NVCC/toolkit below 13.0; runtime capability validation rejects versions below 13000 |
| GPU compute capability | `89` (validated RTX 4080 SUPER) accepted list | required M2 qualification device | CUDA 13.0 runtime 13000, driver API 13010, Windows driver 591.86, 17,170,956,288 device bytes; capability validation rejects devices outside the explicit list |
| No toolkit + `INFERX_ENABLE_CUDA=ON` | — | unsupported | configure failure with toolkit/bootstrap guidance |
| `CMAKE_CUDA_ARCHITECTURES` unset or outside list | — | rejected | configure failure listing the accepted architectures |

CPU presets never detect, include, link, or require CUDA (`INFERX_ENABLE_CUDA=OFF`
default). A CPU sanitizer preset combined with CUDA is rejected at configure
time. CUDA 13.0 is enforced by `find_package(CUDAToolkit 13.0 REQUIRED)` when
enabled. Configure/compile evidence from an older local toolkit is diagnostic
only and cannot close M2. The qualifying CUDA 13.0 WSL2 artifacts are under
`out/benchmarks/m2-cuda13-wsl-final` and
`out/sanitizer/m2-cuda13-wsl-final`, with the clean cross-artifact manifest at
`out/evidence/m2-cuda13-wsl-final/manifest.json`.

## Release channel

Public binary distribution is **blocked** until [ADR 0007](adr/0007-project-licensing.md)
is accepted (see `INTERNAL_USE_ONLY.md`).
