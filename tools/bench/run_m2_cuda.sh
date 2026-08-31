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
source_root="$(pwd -P)"
output="$(realpath -m "$output")"
case "$output" in
  "$source_root"/out/*) ;;
  *) echo "M2 benchmark output must be under $source_root/out" >&2; exit 2 ;;
esac
mkdir -p "$output"
git rev-parse HEAD >"$output/commit.txt"
git status --short >"$output/worktree-status.txt"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
  --format=csv,noheader >"$output/gpu.txt"
nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory,pstate,power.draw \
  --format=csv,noheader >"$output/performance-state.txt" 2>&1 || true
if [[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
  cp /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor "$output/cpu-governor.txt"
fi
nvcc --version >"$output/nvcc.txt"
cmake --version >"$output/cmake.txt"
grep -E 'CMAKE_(CXX|CUDA)_COMPILER(:|_VERSION)|CMAKE_CUDA_ARCHITECTURES' \
  "out/build/$preset/CMakeCache.txt" >"$output/build-config.txt"
device_info="out/build/$preset/inferx-device-info"
if [[ ! -x "$device_info" ]]; then
  echo "missing M2 device diagnostic: $device_info" >&2
  exit 1
fi
"$device_info" --json >"$output/device-info.json"
runner=()
if command -v taskset >/dev/null 2>&1 && taskset -c 0 true >/dev/null 2>&1; then
  runner=(taskset -c 0)
fi
for benchmark in inferx_memory_pool_benchmark inferx_cuda_copy_benchmark \
                 inferx_cuda_event_benchmark inferx_cuda_launch_benchmark; do
  binary="out/build/$preset/benchmarks/$benchmark"
  if [[ ! -x "$binary" ]]; then
    echo "missing M2 benchmark: $binary" >&2
    exit 1
  fi
  "${runner[@]}" "$binary" --benchmark_format=json --benchmark_repetitions=30 \
    --benchmark_out="$output/$benchmark.json"
done
python3 tools/bench/check_m2_overhead.py \
  --copy "$output/inferx_cuda_copy_benchmark.json" \
  --event "$output/inferx_cuda_event_benchmark.json" \
  --launch "$output/inferx_cuda_launch_benchmark.json" \
  --output "$output/overhead.json"
sha256sum "$output"/*.json >"$output/sha256.txt"
