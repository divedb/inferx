#!/usr/bin/env bash
set -euo pipefail

preset=""
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$preset" || -z "$output" ]]; then
  echo "usage: $0 --preset PRESET --output DIR" >&2
  exit 2
fi
mkdir -p "$output"
git rev-parse HEAD >"$output/commit.txt"
git status --short >"$output/worktree-status.txt"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
  --format=csv,noheader >"$output/gpu.txt"
nvcc --version >"$output/nvcc.txt"
cmake --version >"$output/cmake.txt"
grep -E 'CMAKE_(CXX|CUDA)_COMPILER(:|_VERSION)|CMAKE_CUDA_ARCHITECTURES' \
  "out/build/$preset/CMakeCache.txt" >"$output/build-config.txt"
for benchmark in inferx_memory_pool_benchmark inferx_cuda_copy_benchmark \
                 inferx_cuda_event_benchmark inferx_cuda_launch_benchmark; do
  binary="out/build/$preset/benchmarks/$benchmark"
  if [[ ! -x "$binary" ]]; then
    echo "missing M2 benchmark: $binary" >&2
    exit 1
  fi
  "$binary" --benchmark_format=json --benchmark_repetitions=30 \
    --benchmark_out="$output/$benchmark.json"
done
python3 tools/bench/check_m2_overhead.py \
  --copy "$output/inferx_cuda_copy_benchmark.json" \
  --event "$output/inferx_cuda_event_benchmark.json" \
  --launch "$output/inferx_cuda_launch_benchmark.json" \
  --output "$output/overhead.json"
sha256sum "$output"/*.json >"$output/sha256.txt"
