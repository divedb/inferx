#!/usr/bin/env python3
"""CLI contract for local/cache/Hub model naming in inferx-model-inspect."""

import argparse
import subprocess
import sys
import tempfile


def run(binary: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, *arguments], capture_output=True, text=True, timeout=60, check=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--expect-downloads-disabled", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    result = run(args.binary, "--help")
    check(result.returncode == 0, "--help exits 0")
    check("MODEL [--revision" in result.stdout, "help documents positional MODEL")
    check("Hugging Face model ID" in result.stdout, "help documents Hub IDs")

    result = run(args.binary, args.fixture, "--offline")
    check(result.returncode == 0, "local fixture resolves while offline")
    check("model_source=local\n" in result.stdout, "local source is reported")
    check("architecture=llama\n" in result.stdout, "resolved local model is inspected")

    with tempfile.TemporaryDirectory(prefix="inferx-cli-cache-") as cache:
        result = run(
            args.binary,
            "Qwen/Qwen2-test",
            "--offline",
            "--download-dir",
            cache,
        )
    check(result.returncode == 1, "offline cache miss exits 1")
    check("model resolution failed" in result.stderr, "resolution error is scoped")
    check("offline operation was requested" in result.stderr, "offline miss is actionable")

    if args.expect_downloads_disabled:
        with tempfile.TemporaryDirectory(prefix="inferx-cli-cache-") as cache:
            result = run(args.binary, "Qwen/Qwen2-test", "--download-dir", cache)
        check(result.returncode == 1, "Hub-disabled cache miss exits 1")
        check(
            "Hugging Face downloads disabled" in result.stderr,
            "Hub-disabled build reports how to enable downloads",
        )

    result = run(args.binary, "./missing-model")
    check(result.returncode == 1, "missing explicit path exits 1")
    check("looks like a local path" in result.stderr, "path typo is not reported as Hub failure")

    result = run(args.binary, args.fixture, "--unknown")
    check(result.returncode == 2, "unknown option exits 2")
    check("unknown option" in result.stderr, "unknown option is named")
    check("usage:" in result.stderr, "usage follows argument error")

    if failures:
        for failure in failures:
            print(f"model inspect CLI FAIL: {failure}", file=sys.stderr)
        return 1
    print("model inspect CLI ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
