#!/usr/bin/env python3
"""Expected-failure test driver (m0.md section 13.5).

A test passes only when the command FAILS **and** every expected message
fragment appears in its output; an unrelated failure does not satisfy it.

Modes:
  --fixture FILE [--arg K=V ...] [--build]
      Copies FILE as a standalone CMake project into the work directory,
      configures it (with -D args), and requires configure failure — or, with
      --build, requires configure success followed by build failure.
  --script FILE [--arg K=V ...]
      Runs an executable script and requires a nonzero exit with the
      fragments in its combined output.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expect", action="append", default=[],
                        help="message fragment that must appear in the failing output")
    parser.add_argument("--fixture", type=Path, help="CMakeLists-style fixture file")
    parser.add_argument("--script", type=Path, help="executable script fixture")
    parser.add_argument("--arg", action="append", default=[],
                        help="K=V passed through (as -DK=V for fixtures, --K V for scripts)")
    parser.add_argument("--build", action="store_true",
                        help="configure must succeed and the build must fail")
    args = parser.parse_args()

    if not args.expect:
        sys.exit("error: at least one --expect fragment is required")
    if bool(args.fixture) == bool(args.script):
        sys.exit("error: exactly one of --fixture or --script is required")

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)

    if args.script:
        import os
        command = [str(args.script)]
        if not os.access(args.script, os.X_OK):
            command = [sys.executable] + command
        for pair in args.arg:
            key, _, value = pair.partition("=")
            command += [f"--{key.replace('_', '-')}", value]
    else:
        project_dir = args.work_dir / "project"
        project_dir.mkdir()
        shutil.copyfile(args.fixture, project_dir / "CMakeLists.txt")
        command = ["cmake", "-S", str(project_dir), "-B", str(project_dir / "build"),
                   "-G", "Ninja"]
        for pair in args.arg:
            command.append(f"-D{pair}")

    print(f"+ {' '.join(command)}")
    configure = subprocess.run(command, capture_output=True, text=True, timeout=240)

    if args.script:
        output = configure.stdout + configure.stderr
        if configure.returncode == 0:
            print(output)
            sys.exit("error: script unexpectedly succeeded; an expected-failure test "
                     "requires a nonzero exit")
    elif args.build:
        if configure.returncode != 0:
            print(configure.stdout + configure.stderr)
            sys.exit("error: configure unexpectedly failed; --build fixtures must "
                     "configure cleanly and fail at build time")
        build = subprocess.run(
            ["cmake", "--build", str(args.work_dir / "project" / "build"), "--parallel"],
            capture_output=True, text=True, timeout=240)
        output = build.stdout + build.stderr
        if build.returncode == 0:
            print(output)
            sys.exit("error: build unexpectedly succeeded; an expected-failure test "
                     "requires a failing build")
    else:
        output = configure.stdout + configure.stderr
        if configure.returncode == 0:
            print(output)
            sys.exit("error: configure unexpectedly succeeded; an expected-failure "
                     "test requires a failing configure")

    missing = [fragment for fragment in args.expect if fragment not in output]
    if missing:
        print(output)
        sys.exit("error: command failed for an UNEXPECTED reason; missing fragments: "
                 f"{missing}")
    print("expected failure reproduced with the expected message")
    return 0


if __name__ == "__main__":
    sys.exit(main())
