#!/usr/bin/env python3
"""Reject CUDA SDK headers from hardware-neutral InferX source trees."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


FORBIDDEN = re.compile(
    r"^\s*#\s*include\s*[<\"](?:cuda(?:_runtime[^/>]*)?|cublas[^/>]*|nccl)\.h[>\"]",
    re.MULTILINE,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    roots = ["include/inferx", "src", "simulator", "tests/unit/tensor", "tests/unit/runtime"]
    failures: list[str] = []
    for relative in roots:
        root = args.source_root / relative
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".h", ".cc", ".cpp", ".cxx"}:
                continue
            if FORBIDDEN.search(path.read_text(encoding="utf-8")):
                failures.append(str(path.relative_to(args.source_root)))
    if failures:
        print("error: CUDA SDK headers crossed hardware-neutral boundaries:", file=sys.stderr)
        for path in failures:
            print(f"- {path}", file=sys.stderr)
        return 1
    print("CUDA header boundary: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
