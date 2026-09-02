#!/usr/bin/env bash
# Format check/fix for InferX-owned sources.
#
# The file list comes from `git ls-files` limited to InferX-owned extensions,
# so third_party (gitlinks, never recursed) and out/ (ignored, untracked) are
# excluded by construction — not by a broad find filter.
#
# Usage:
#   tools/check_format.sh            # check only (CI mode: never mutates)
#   tools/check_format.sh --fix      # developer mode: rewrite tracked files
#
# The clang-format major version is pinned (formatting output changes between
# releases); override the binary with INFERX_CLANG_FORMAT for local lanes.
set -euo pipefail

PINNED_MAJOR=18
MODE=check
if [[ "${1:-}" == "--fix" ]]; then
  MODE=fix
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--fix]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FORMAT="${INFERX_CLANG_FORMAT:-}"
if [[ -z "$FORMAT" ]]; then
  for candidate in "clang-format-$PINNED_MAJOR" clang-format; do
    if command -v "$candidate" >/dev/null 2>&1; then
      FORMAT="$candidate"
      break
    fi
  done
fi
if [[ -z "$FORMAT" ]]; then
  echo "error: clang-format (major $PINNED_MAJOR) not found; install clang-format-$PINNED_MAJOR or set INFERX_CLANG_FORMAT" >&2
  exit 1
fi

ACTUAL_MAJOR="$("$FORMAT" --version | sed -E 's/[^0-9]*([0-9]+)\..*/\1/')"
if [[ "$ACTUAL_MAJOR" != "$PINNED_MAJOR" ]]; then
  echo "error: pinned clang-format major is $PINNED_MAJOR but '$FORMAT' is $ACTUAL_MAJOR; formatting output differs between releases (see docs/development.md)" >&2
  exit 1
fi

mapfile -t FILES < <(git ls-files -- '*.h' '*.cc' '*.cu' '*.cuh')
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "format: no tracked InferX sources yet"
  exit 0
fi

if [[ "$MODE" == "fix" ]]; then
  exec "$FORMAT" -i "${FILES[@]}"
fi

if ! "$FORMAT" --dry-run --Werror "${FILES[@]}" >/dev/null 2>&1; then
  "$FORMAT" --dry-run --Werror "${FILES[@]}" 2>&1 | head -50 || true
  echo "error: format check failed; run tools/check_format.sh --fix (or the 'format' target) with clang-format $PINNED_MAJOR" >&2
  exit 1
fi
echo "format: ${#FILES[@]} tracked sources conform (clang-format $PINNED_MAJOR)"
