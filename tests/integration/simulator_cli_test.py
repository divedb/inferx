#!/usr/bin/env python3
"""End-to-end inferx simulate command and stable exit-code coverage."""

import argparse
import pathlib
import subprocess
import tempfile


def run(command: list[str], expected: int) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != expected:
        raise AssertionError(
            f"expected exit {expected}, got {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workload", required=True)
    args = parser.parse_args()

    binary = pathlib.Path(args.binary)
    with tempfile.TemporaryDirectory(prefix="inferx-simulator-cli-") as temporary:
        root = pathlib.Path(temporary)
        trace = root / "trace.jsonl"
        replayed = root / "replayed.jsonl"
        invalid = root / "invalid.json"
        invalid.write_text('{"unknown_field":1}\n', encoding="utf-8")

        prefix = [str(binary), "simulate"]
        run([*prefix, "validate-config", f"--config={args.config}"], 0)
        run([*prefix, "validate-config", f"--config={invalid}"], 2)
        run([*prefix, "validate-config", "--unknown=1"], 2)
        run(
            [
                *prefix,
                "validate-config",
                f"--config={args.config}",
                "--max-active-sequences=not-a-number",
            ],
            2,
        )
        run(
            [
                *prefix,
                "run",
                f"--config={args.config}",
                f"--workload={args.workload}",
                f"--trace={trace}",
            ],
            0,
        )
        run([*prefix, "check-trace", f"--trace={trace}"], 0)
        run(
            [*prefix, "replay", f"--trace={trace}", f"--output={replayed}"],
            0,
        )
        if trace.read_bytes() != replayed.read_bytes():
            raise AssertionError("replayed trace is not byte-identical")
        run(
            [
                *prefix,
                "run",
                f"--config={args.config}",
                f"--workload={args.workload}",
                f"--trace={trace}",
            ],
            4,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
