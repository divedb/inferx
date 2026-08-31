#!/usr/bin/env python3
"""Write the canonical manifest for retained M2 GPU qualification artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], text=True, encoding="utf-8"
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    args = parser.parse_args()

    root = args.source_root.resolve()
    output = args.output.resolve()
    files: list[Path] = []
    for artifact in args.artifact:
        path = (root / artifact).resolve()
        if path == output or not path.exists():
            continue
        if path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
        else:
            files.append(path)
    artifacts = []
    for path in sorted(set(files)):
        data = path.read_bytes()
        artifacts.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    status = git_output("status", "--porcelain=v1", "--untracked-files=no")
    manifest = {
        "schema_version": 1,
        "milestone": "m2",
        "commit": git_output("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
        "runner": {
            "name": os.environ.get("RUNNER_NAME", "unknown"),
            "os": os.environ.get("RUNNER_OS", "unknown"),
            "architecture": os.environ.get("RUNNER_ARCH", "unknown"),
        },
        "outcomes": {
            "tests": os.environ.get("INFERX_M2_TEST_OUTCOME", "unknown"),
            "compute_sanitizer": os.environ.get(
                "INFERX_M2_SANITIZER_OUTCOME", "unknown"
            ),
            "benchmarks": os.environ.get(
                "INFERX_M2_BENCHMARK_OUTCOME", "unknown"
            ),
        },
        "commands": [
            "ctest --preset cuda-release -L <m2-label> --output-on-failure",
            "tools/ci/run_compute_sanitizer.sh --preset cuda-release --suite m2 --output out/sanitizer/m2",
            "tools/bench/run_m2_cuda.sh --preset cuda-release --output out/benchmarks/m2",
        ],
        "seeds": ["cpu-reference-transform-v1", "cuda-strided-copy-v1"],
        "artifacts": artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
