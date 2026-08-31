#!/usr/bin/env python3
"""Fail when an M1 roadmap CTest category is empty."""

import argparse
import json
import pathlib
import subprocess


REQUIRED = (
    "m1-unit",
    "m1-integration",
    "m1-correctness",
    "m1-failure",
    "m1-channel",
    "m1-stress",
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
        if "m1" in labels:
            general += 1
        for label in REQUIRED:
            if label in labels:
                counts[label] += 1
        if any(label in labels for label in REQUIRED) and "m1" not in labels:
            missing_general.append(test["name"])
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        raise SystemExit(f"error: empty M1 CTest labels: {', '.join(missing)}")
    if missing_general:
        raise SystemExit(
            "error: M1 category tests missing general m1 label: "
            + ", ".join(missing_general[:10])
        )
    print("M1 CTest labels:", " ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
