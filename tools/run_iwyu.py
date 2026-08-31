#!/usr/bin/env python3
"""include-what-you-use enforcement over InferX translation units (m0.md
section 9.3).

Replays each InferX-owned compile command from a build directory's
compile_commands.json through include-what-you-use (qualified version 0.21)
and fails on any keep/add/remove violation. Used by the `check-analysis`
target and CI's analysis lane; the analysis preset also surfaces IWYU output
during normal compilation.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_TREES = ("include", "src", "apps", "tests", "benchmarks", "platform")


def project_commands(build_dir: Path, repo_root: Path):
    commands = json.loads(
        (build_dir / "compile_commands.json").read_text(encoding="utf-8"))
    root = repo_root.resolve()
    selected = []
    seen = set()
    for entry in commands:
        path = Path(os.path.abspath(entry["file"])).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        parts = relative.parts
        if "third_party" in parts or "out" in parts:
            continue
        if parts[0] not in PROJECT_TREES:
            continue
        if path.suffix == ".cc" and str(path) not in seen:
            seen.add(str(path))
            selected.append((path, entry["command"]))
    return selected


def to_iwyu_command(command: str, iwyu: str) -> list[str]:
    tokens = shlex.split(command)
    out = [iwyu]
    skip_next = False
    for token in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in ("-o", "-c", "-MF"):
            skip_next = True
            continue
        if token in ("-MMD", "-MD", "-MT", "-MP"):
            continue
        out.append(token)
    out += ["-Xiwyu", "--no_fwd_decls"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True,
                        help="build tree containing compile_commands.json")
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[1],
                        help="repository root (default: derived from this script)")
    parser.add_argument("--iwyu", default=None,
                        help="include-what-you-use binary (default: autodetect)")
    args = parser.parse_args()

    if not (args.build_dir / "compile_commands.json").is_file():
        sys.exit("error: no compile_commands.json; run the analysis preset first")

    iwyu = args.iwyu
    if iwyu is None:
        for candidate in ("include-what-you-use", "iwyu"):
            probe = subprocess.run(["sh", "-c", f"command -v {candidate}"],
                                   capture_output=True, text=True)
            if probe.returncode == 0:
                iwyu = candidate
                break
    if iwyu is None:
        sys.exit("error: include-what-you-use (qualified: 0.21) not found")

    selected = project_commands(args.build_dir, args.source_root)
    if not selected:
        sys.exit("error: no InferX translation units found in compile_commands.json")

    violations = 0
    for path, command in selected:
        result = subprocess.run(to_iwyu_command(command, iwyu),
                                capture_output=True, text=True, timeout=300)
        diagnostics = [line for line in (result.stdout + result.stderr).splitlines()
                       if line.startswith(("\t", "should add", "should remove"))]
        if result.returncode != 0 and diagnostics:
            violations += 1
            print(f"IWYU violations in {path}:")
            for line in diagnostics[:20]:
                print(f"  {line}")

    if violations:
        sys.exit(f"error: {violations} translation unit(s) with include-what-you-use "
                 "violations")
    print(f"iwyu: {len(selected)} InferX translation units clean ({iwyu})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
