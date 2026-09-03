#!/usr/bin/env python3
"""Process-level contract tests for the unified inferx CLI."""

import argparse
import subprocess
import sys


def run(binary: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, *args], capture_output=True, text=True, timeout=60, check=False
    )


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
    check(
        result.stdout == f"inferx {args.expected_version}\n",
        f"--version is stable (got {result.stdout!r})",
    )

    result = run(args.binary, "version")
    check(result.returncode == 0, "version exits 0")
    for prefix in (
        "inferx:",
        "git-revision:",
        "compiler:",
        "c++-standard:",
        "build-type:",
        "features:",
    ):
        check(
            any(line.startswith(prefix) for line in result.stdout.splitlines()),
            f"version prints {prefix}",
        )
    check("c++-standard: 23" in result.stdout, "version reports C++23")

    result = run(args.binary, "env")
    check(result.returncode == 0, "env exits 0 without a GPU")
    check("compiled.cuda=" in result.stdout, "env reports compiled CUDA support")
    check("cuda.device_count=" in result.stdout, "env reports visible devices")

    result = run(args.binary, "--help")
    check(result.returncode == 0, "--help exits 0")
    check("InferX unified inference runtime" in result.stdout, "top-level help has purpose")
    for command in (
        "serve",
        "bench",
        "run",
        "chat",
        "complete",
        "inspect",
        "download",
        "version",
        "env",
    ):
        check(command in result.stdout, f"top-level help lists {command}")

    for command in (
        ("serve",),
        ("bench",),
        ("bench", "latency"),
        ("run",),
        ("chat",),
        ("complete",),
        ("inspect",),
        ("download",),
        ("version",),
        ("env",),
    ):
        result = run(args.binary, *command, "--help")
        check(result.returncode == 0, f"{' '.join(command)} --help exits 0")

    result = run(args.binary)
    check(result.returncode == 2, "no subcommand exits 2")
    check("InferX unified inference runtime" in result.stdout, "no subcommand prints help")

    result = run(args.binary, "does-not-exist")
    check(result.returncode == 2, "unknown command exits 2")
    check(result.stderr.startswith("error: inferx:"), "unknown command has error prefix")
    check("does-not-exist" in result.stderr, "unknown command is named")

    result = run(args.binary, "serve")
    check(result.returncode == 2, "missing --model exits 2")
    check(result.stderr.startswith("error: inferx:"), "missing argument has error prefix")
    check("--model" in result.stderr, "missing required argument is named")

    result = run(args.binary, "serve", "--model", "x", "--dtype", "complex128")
    check(result.returncode == 2, "invalid enum exits 2")
    check("--dtype" in result.stderr, "invalid enum identifies --dtype")

    result = run(args.binary, "serve", "--model", "x", "--top-p", "0")
    check(result.returncode == 2, "invalid top-p exits 2")

    result = run(args.binary, "chat", "--interactive", "--prompt", "hello")
    check(result.returncode == 2, "mutually exclusive chat modes exit 2")

    result = run(
        args.binary,
        "--log-level",
        "debug",
        "--seed",
        "42",
        "serve",
        "--model",
        "x",
    )
    check(result.returncode == 6, "valid unavailable serve command exits 6")
    check("feature unavailable" in result.stderr, "unavailable feature is explicit")

    result = run(
        args.binary,
        "bench",
        "latency",
        "--model",
        "x",
        "--output-format",
        "json",
    )
    check(result.returncode == 6, "valid unavailable benchmark exits 6")

    if failures:
        for failure in failures:
            print(f"cli test FAIL: {failure}", file=sys.stderr)
        return 1
    print("cli test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
