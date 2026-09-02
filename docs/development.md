# InferX development guide

How to build, test, and investigate InferX locally and in containers. The
commands below are the same ones CI runs (`docs/milestones/m0.md` section 15
is the acceptance contract; this guide matches it). Platform support:
[`supported-platforms.md`](supported-platforms.md). Dependencies:
[`dependency-policy.md`](dependency-policy.md).

## Prerequisites (CPU only)

Ubuntu 24.04 with: GCC 13 (`g++-13`), Clang 18 (`clang-18 clang-tidy-18
clang-format-18`), CMake ≥ 3.28, Ninja ≥ 1.11, Python 3.12, `include-what-you-use`
(for the analysis lane), and OpenSSL 3 development files for native Hugging Face downloads. CUDA is
optional.

One-time initialization (the only network step; everything after is offline):

```bash
python3 tools/deps/bootstrap.py --profile core   # Abseil, simdjson, BLAKE3, tests/benchmark
python3 tools/deps/bootstrap.py --list           # see profiles and statuses
```

## Daily loop

```bash
cmake --preset dev-clang
cmake --build --preset dev-clang --parallel
ctest --preset dev-clang
cmake --build --preset dev-clang --target check   # fast + analysis + integration gates
```

Presets (all Ninja, build trees under `out/build/<preset>`; the same name is
the build/test preset):

| Preset | Purpose |
|---|---|
| `dev-gcc` / `dev-clang` | Debug development, warnings-as-errors |
| `asan-ubsan` / `tsan` | Clang Debug sanitizer lanes (mutually exclusive by design) |
| `cpu-release` | Release lane used by install/consumer and metrics |
| `analysis` | clang-tidy + IWYU during compilation |
| `cuda-release` | CUDA smoke; explicit `CMAKE_CUDA_ARCHITECTURES` (never `native`) |

Machine-specific overrides (CUDA toolkit path, extra architectures) go in
`CMakeUserPresets.json` — gitignored on purpose.

## Quality gates

```bash
tools/check_format.sh              # format-check (CI mode, never mutates)
tools/check_format.sh --fix        # or: cmake --build out/build/dev-gcc --target format
tools/run_clang_tidy.sh out/build/analysis
python3 tools/run_iwyu.py --build-dir out/build/analysis
python3 tools/check_headers.py --source-root . --build-dir out/build/analysis
python3 tools/deps/check_manifest.py        # dependency drift gate
```

Aggregate targets: `check-fast` (format + unit/compile), `check-analysis`
(header probes + tidy + IWYU), `check` (all CPU gates in this tree). Sanitizer
and CUDA lanes stay separate invocations; aggregates never configure other
trees.

Notes:

- clang-format is pinned to major **18**; the pinned version cannot spell
  `Standard: c++23` (needs 19+), so `.clang-format` uses `Latest` — see the
  comment there and [supported-platforms.md](supported-platforms.md).
- `std::expected` is unavailable on Clang 18 + libstdc++ 13 (documented gap);
  the C++23 sentinel covers accepted alternatives on every lane.
- Suppressions (`NOLINT`) must be local, name the check, and say why.

## Install and downstream consumption

```bash
cmake --preset cpu-release && cmake --build --preset cpu-release --parallel
ctest --test-dir out/build/cpu-release -L integration --output-on-failure
```

The `install_consumer` integration test installs to a temporary prefix and
configures `tests/consumer` with only `CMAKE_PREFIX_PATH`; `install_absl_consumer`
does the same for the pinned Abseil package (the M1 installed-dependency
strategy proof).

## CUDA (optional)

A developer without CUDA can complete every CPU task. With a toolkit:

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release --parallel
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
# (the M2 inferx-device-info tool was removed with the platform substrate)
ctest --preset cuda-release -L m2 --output-on-failure
tools/ci/run_compute_sanitizer.sh --preset cuda-release --suite m2 \
  --output out/sanitizer/m2
tools/bench/run_m2_cuda.sh --preset cuda-release --output out/benchmarks/m2
```

- Architectures are explicit (default accepted list: `89`); override in
  `CMakeUserPresets.json`, never via `native` in shared presets.
- No GPU visible? CUDA executables report a failure locally; the owned runner
  additionally requires every M2 label to select and pass tests without skips.
- `INFERX_ENABLE_CUDA=ON` with a missing toolkit is a configure error, never a
  silent CPU fallback (`tests/failure/` covers it).

## CLI diagnostics

```bash
out/build/dev-clang/apps/inferx_info/inferx-info --version   # stable single line
out/build/dev-clang/apps/inferx_info/inferx-info --build     # compiler/std/features/architectures
```

M1 simulator/replay checks:

```bash
out/build/dev-clang/inferx-sim validate-config --config tests/integration/simulator/data/basic_config.json
out/build/dev-clang/inferx-sim run --config tests/integration/simulator/data/basic_config.json \
  --workload tests/integration/simulator/data/basic_workload.json --trace out/traces/basic.jsonl
out/build/dev-clang/inferx-sim replay --trace out/traces/basic.jsonl \
  --output out/traces/basic.replayed.jsonl
ctest --preset dev-clang -L 'm1-unit|m1-integration|m1-correctness|m1-failure|m1-stress'
```

See [scheduler.md](scheduler.md), [simulator.md](simulator.md), and
[replay-schema.md](replay-schema.md) for the contracts and stable CLI exits.

## Metrics, stress, and SBOM evidence

```bash
python3 tools/ci/measure_build.py --preset cpu-release --output out/metrics/m0.json
python3 tools/ci/repeat_build.py --jobs "$(nproc)"           # 3 fresh flows under out/stress/
out/build/asan-ubsan/inferx_simulator_stress --seed=0x4d31535452455353 \
  --operations=1000000 --failure_output=out/failures/m1
out/build/cpu-release/inferx_scheduler_benchmark --benchmark_format=json \
  --benchmark_out=out/metrics/m1-scheduler.json
tools/deps/generate_sbom.sh out/install/cpu-release out/sbom/inferx-m0.spdx.json
```

The SBOM script uses a pinned Syft release (`v1.51.1`, sha256-recorded),
writes only below the output path plus the user cache, validates the JSON, and
never uploads. Metrics JSON is required evidence for the CI `build-metrics`
job; M0 has no time gates.

## Containers

```bash
docker build -f docker/cpu-dev.Dockerfile  -t inferx-cpu-dev  .   # pinned ubuntu:24.04 digest
docker build -f docker/cuda-dev.Dockerfile -t inferx-cuda-dev .   # pinned CUDA 13.0 devel digest

docker run --rm -it -v "$PWD:/workspace" -w /workspace inferx-cpu-dev \
  bash -lc 'python3 tools/deps/bootstrap.py --profile core && \
            cmake --preset dev-gcc && cmake --build --preset dev-gcc --parallel && ctest --preset dev-gcc'
```

Images run as non-root `inferx` with UTF-8 locale; no credentials or host
paths are baked in; project dependencies are never downloaded into images.

## Troubleshooting

- **Missing submodule configure error**: run the exact `bootstrap.py --profile`
  command in the message; CMake never downloads.
- **Manifest drift**: `python3 tools/deps/check_manifest.py` names the path and
  both revisions; fix gitlink + manifest together.
- **CUDA host-compiler/toolkit mismatch**: see the pairing table in
  [supported-platforms.md](supported-platforms.md). M2 rejects CUDA 12.x; use
  the pinned CUDA 13.0 image or an equivalent newer toolkit. CMake isolates
  `.cu` at C++20 when the qualified combination does not expose `cuda_std_23`
  (ADR 0020).
- **clang-tidy/IWYU not found in analysis preset**: install `clang-tidy-18` /
  `include-what-you-use` or point `INFERX_CLANG_TIDY_BIN` / `INFERX_IWYU_BIN`.
- **Clean rebuild**: remove `out/build/<preset>` (build-directory-specific —
  never a broad source-tree deletion).

## Disk-space expectations

A full core bootstrap plus all CPU presets uses roughly 1–2 GiB under
`third_party/` and `out/`. `out/` is entirely disposable; delete any subtree
without ceremony.
