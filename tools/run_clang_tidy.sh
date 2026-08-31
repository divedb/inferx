#!/usr/bin/env bash
# clang-tidy over InferX compile commands (m0.md section 9.2).
#
# Requires an existing 'analysis' configure directory (or any build dir with
# compile_commands.json given as $1). Selects project files only — dependency
# and generated files are excluded — runs the qualified clang-tidy binary,
# and keeps warnings-as-errors. Fixes are emitted only in the explicitly
# requested developer mode (--fix), never in CI.
set -euo pipefail

PINNED_MAJOR=18
FIX_MODE=off
BUILD_DIR="${1:-}"
if [[ "${1:-}" == "--fix" ]]; then
  FIX_MODE=on
  BUILD_DIR="${2:-}"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "$BUILD_DIR" ]]; then
  BUILD_DIR="$ROOT/out/build/analysis"
fi
if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
  echo "error: no compile_commands.json in '$BUILD_DIR'; configure first: cmake --preset analysis" >&2
  exit 1
fi

TIDY="${INFERX_CLANG_TIDY:-}"
if [[ -z "$TIDY" ]]; then
  for candidate in "clang-tidy-$PINNED_MAJOR" clang-tidy; do
    if command -v "$candidate" >/dev/null 2>&1; then
      TIDY="$candidate"
      break
    fi
  done
fi
if [[ -z "$TIDY" ]]; then
  echo "error: clang-tidy (major $PINNED_MAJOR) not found; install clang-tidy-$PINNED_MAJOR or set INFERX_CLANG_TIDY" >&2
  exit 1
fi

# Project files only: absolute paths under the repo, in InferX-owned trees,
# excluding third_party and everything under out/.
mapfile -t FILES < <(python3 - "$BUILD_DIR" "$ROOT" <<'EOF'
import json
import os
import sys

build_dir = os.path.abspath(sys.argv[1])
repo_root = os.path.abspath(sys.argv[2])
commands = json.load(open(os.path.join(build_dir, "compile_commands.json")))
seen = set()
for entry in commands:
    path = os.path.abspath(entry["file"])
    if not path.startswith(repo_root + os.sep):
        continue
    relative = os.path.relpath(path, repo_root)
    parts = relative.split(os.sep)
    if "third_party" in parts or "out" in parts:
        continue
    if not any(parts[0] == tree for tree in ("include", "src", "apps", "tests",
                                             "benchmarks", "platform")):
        continue
    if path.endswith(".cc") and path not in seen:
        seen.add(path)
        print(path)
EOF
)
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "clang-tidy: no InferX translation units in $BUILD_DIR" >&2
  exit 1
fi

ARGS=(--config-file "$ROOT/.clang-tidy" -p "$BUILD_DIR")
if [[ "$FIX_MODE" == "on" ]]; then
  ARGS+=(--fix)
fi

# Warnings stay errors (.clang-tidy: WarningsAsErrors '*'); a nonzero tidy
# exit propagates. Suppressions must be local, name the check, and state why.
"$TIDY" "${ARGS[@]}" "${FILES[@]}"
echo "clang-tidy: ${#FILES[@]} InferX translation units clean"
