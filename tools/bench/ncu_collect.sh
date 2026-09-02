#!/usr/bin/env bash
# NCU collection for the InferX-vs-TokenSpeed comparison (ADR 0032
# addendum): profiles both implementations on the shared workload matrix
# and exports per-kernel CSV metric reports.
#
# Usage: tools/bench/ncu_collect.sh [output_dir]
# Requires: CUDA 13 ncu on PATH or /usr/local/cuda-13.0/bin, the built
# inferx_kernels_benchmark, and the bench venv with tokenspeed deps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/docs/benchmarks/ncu}"
NCU="${NCU:-/usr/local/cuda-13.0/bin/ncu}"
mkdir -p "$OUT"

METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__occupancy_limit_registers"

echo "== InferX benchmark under NCU =="
"$NCU" --target-processes all \
  --metrics "$METRICS" \
  --csv --page raw \
  --launch-count "${NCU_LAUNCH_COUNT:-200}" \
  -o "$OUT/inferx" -f \
  "$ROOT/out/build/cuda-kernels/benchmarks/inferx_kernels_benchmark" \
  > "$OUT/inferx_stdout.txt" 2>&1 || echo "ncu inferx rc=$? (see $OUT/inferx_stdout.txt)"

echo "== TokenSpeed benchmark under NCU (subset: kernels only) =="
"$NCU" --target-processes all \
  --metrics "$METRICS" \
  --csv --page raw \
  --launch-count "${NCU_LAUNCH_COUNT:-200}" \
  --kernel-name-base demangled \
  -o "$OUT/tokenspeed" -f \
  env PYTHONPATH="$ROOT/tools/bench/stubs:$ROOT/tools/bench/tokenspeed/tokenspeed-kernel/python" \
  CUDA_HOME=/usr/local/cuda-13.0 \
  "$ROOT/bench-venv/bin/python" "$ROOT/tools/bench/run_tokenspeed_bench.py" \
  > "$OUT/tokenspeed_stdout.txt" 2>&1 || echo "ncu tokenspeed rc=$? (see $OUT/tokenspeed_stdout.txt)"

# Export the .ncu-rep files to CSV next to the reports.
for name in inferx tokenspeed; do
  if [ -f "$OUT/$name.ncu-rep" ]; then
    "$NCU" --import "$OUT/$name.ncu-rep" --csv --page raw > "$OUT/$name.csv" || true
    echo "wrote $OUT/$name.csv"
  fi
done
echo "done: $OUT"
