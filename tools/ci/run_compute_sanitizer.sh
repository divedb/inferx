#!/usr/bin/env bash
set -euo pipefail

preset=""
build_dir=""
suite=""
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="$2"; shift 2 ;;
    --build-dir) build_dir="$2"; shift 2 ;;
    --suite) suite="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$preset" && -n "$build_dir" ]] ||
   [[ -z "$preset" && -z "$build_dir" ]] ||
   [[ "$suite" != "m2" && "$suite" != "m4" ]] || [[ -z "$output" ]]; then
  echo "usage: $0 (--preset PRESET | --build-dir DIR) --suite m2|m4 --output DIR" >&2
  exit 2
fi
if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo "compute-sanitizer is required for $suite qualification" >&2
  exit 1
fi

mkdir -p "$output"
if [[ -z "$build_dir" ]]; then
  build_dir="out/build/$preset"
fi
binary="$build_dir/inferx-device-info"
if [[ "$suite" == "m4" ]]; then
  binary="$build_dir/tests/inferx_m4_cuda_ops_test"
  if [[ ! -x "$binary" ]]; then
    echo "missing M4 CUDA test binary: $binary" >&2
    exit 1
  fi
else
if [[ ! -x "$binary" ]]; then
  echo "missing M2 diagnostic binary: $binary" >&2
  exit 1
fi
correctness="$build_dir/tests/inferx_tensor_cuda_correctness"
if [[ ! -x "$correctness" ]]; then
  echo "missing M2 CUDA correctness binary: $correctness" >&2
  exit 1
fi
fi

compute-sanitizer --version >"$output/version.txt" 2>&1
run_case() {
  local tool="$1"
  local case_name="$2"
  local executable="$3"
  shift 3
  local log="$output/$tool-$case_name.log"
  local -a command=(compute-sanitizer --tool "$tool" --error-exitcode 86)
  if [[ "$tool" == "memcheck" ]]; then
    command+=(--leak-check full)
  fi
  printf '%q ' "${command[@]}" "$executable" "$@" \
    >"$output/$tool-$case_name.command"
  printf '\n' >>"$output/$tool-$case_name.command"
  timeout 20m "${command[@]}" "$executable" "$@" >"$log" 2>&1
  printf '0\n' >"$output/$tool-$case_name.exit-code"
  if grep -E "ERROR SUMMARY: [1-9]|========= ERROR:|========= WARNING:" "$log"; then
    echo "$tool/$case_name reported an InferX sanitizer error or warning" >&2
    exit 1
  fi
}

if [[ "$suite" == "m2" ]]; then
  run_case memcheck self-test "$binary" --self-test
  INFERX_M2_CUDA_CORRECTNESS_CASES=1000 \
    run_case memcheck correctness "$correctness"
  run_case racecheck synchronization "$binary" --self-test
  run_case initcheck initialized-input "$binary" --self-test
  run_case synccheck synchronization "$binary" --self-test
  cases='["memcheck/self-test","memcheck/correctness","racecheck/synchronization","initcheck/initialized-input","synccheck/synchronization"]'
else
  run_case memcheck cuda-ops "$binary"
  run_case racecheck cuda-ops "$binary"
  run_case initcheck cuda-ops "$binary"
  run_case synccheck cuda-ops "$binary"
  cases='["memcheck/cuda-ops","racecheck/cuda-ops","initcheck/cuda-ops","synccheck/cuda-ops"]'
fi

printf '{"schema_version":1,"suite":"%s","cases":%s,"all_exit_codes":0}\n' \
  "$suite" "$cases" >"$output/manifest.json"
