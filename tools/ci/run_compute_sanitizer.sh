#!/usr/bin/env bash
set -euo pipefail

preset=""
suite=""
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="$2"; shift 2 ;;
    --suite) suite="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$preset" || "$suite" != "m2" || -z "$output" ]]; then
  echo "usage: $0 --preset PRESET --suite m2 --output DIR" >&2
  exit 2
fi
if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo "compute-sanitizer is required for M2 qualification" >&2
  exit 1
fi

mkdir -p "$output"
binary="out/build/$preset/inferx-device-info"
if [[ ! -x "$binary" ]]; then
  echo "missing M2 diagnostic binary: $binary" >&2
  exit 1
fi
correctness="out/build/$preset/tests/inferx_tensor_cuda_correctness"
if [[ ! -x "$correctness" ]]; then
  echo "missing M2 CUDA correctness binary: $correctness" >&2
  exit 1
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

run_case memcheck self-test "$binary" --self-test
INFERX_M2_CUDA_CORRECTNESS_CASES=1000 \
  run_case memcheck correctness "$correctness"
run_case racecheck synchronization "$binary" --self-test
run_case initcheck initialized-input "$binary" --self-test
run_case synccheck synchronization "$binary" --self-test

printf '{"schema_version":1,"suite":"m2","cases":["memcheck/self-test","memcheck/correctness","racecheck/synchronization","initcheck/initialized-input","synccheck/synchronization"],"all_exit_codes":0}\n' \
  >"$output/manifest.json"
