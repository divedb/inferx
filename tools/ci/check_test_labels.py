#!/usr/bin/env python3
"""Fail when a required core CTest category is empty."""

import argparse
import json
import pathlib
import subprocess


REQUIRED = (
    "core-unit",
    "core-channel",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    completed = subprocess.run(
        ["ctest", "--test-dir", str(args.build_dir), "--show-only=json-v1"],
        text=True,
        capture_output=True,
        check=True,
    )
    tests = json.loads(completed.stdout)["tests"]
    counts = {label: 0 for label in REQUIRED}
    general = 0
    missing_general = []
    for test in tests:
        labels = next(
            (prop["value"] for prop in test.get("properties", []) if prop["name"] == "LABELS"),
            [],
        )
        if "core" in labels:
            general += 1
        for label in REQUIRED:
            if label in labels:
                counts[label] += 1
        if any(label in labels for label in REQUIRED) and "core" not in labels:
            missing_general.append(test["name"])
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        raise SystemExit(f"error: empty core CTest labels: {', '.join(missing)}")
    if missing_general:
        raise SystemExit(
            "error: core category tests missing general core label: "
            + ", ".join(missing_general[:10])
        )
    print("Core CTest labels:", " ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
