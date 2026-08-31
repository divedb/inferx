#!/usr/bin/env python3
"""CLI behavior test for inferx-info (label: integration).

Verifies documented options, stable --version output, --build metadata, and
stable exit codes for invalid arguments.
"""

import argparse
import subprocess
import sys


def run(binary: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([binary, *args], capture_output=True, text=True, timeout=60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            failures.append(label)

    result = run(args.binary, "--version")
    check(result.returncode == 0, "--version exits 0")
    check(result.stdout == f"inferx-info {args.expected_version}\n",
          f"--version is the stable single line (got {result.stdout!r})")

    result = run(args.binary, "--build")
    check(result.returncode == 0, "--build exits 0")
    for prefix in ("inferx:", "compiler:", "c++-standard:", "build-type:", "features:"):
        check(any(line.startswith(prefix) for line in result.stdout.splitlines()),
              f"--build prints {prefix}")
    check(f"c++-standard: 23" in result.stdout, "--build reports C++23")

    result = run(args.binary, "--help")
    check(result.returncode == 0, "--help exits 0")
    check("usage: inferx-info" in result.stdout, "--help prints usage")

    result = run(args.binary)
    check(result.returncode == 0, "no options exits 0 with usage")

    result = run(args.binary, "--bogus")
    check(result.returncode == 2, "unknown option exits 2")
    check("unknown option" in result.stderr, "unknown option message on stderr")
    check("usage:" in result.stderr, "usage follows unknown option")

    result = run(args.binary, "--version", "--build")
    check(result.returncode == 2, "multiple options exit 2")

    if failures:
        for failure in failures:
            print(f"cli test FAIL: {failure}", file=sys.stderr)
        return 1
    print("cli test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
