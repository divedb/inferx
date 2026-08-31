#!/usr/bin/env python3
"""Validate the M1 scheduler JSON latency/allocation acceptance gate."""

import argparse
import json
import pathlib


def nanoseconds(value: float, unit: str) -> float:
    scales = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}
    return value * scales[unit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    expected = ("Prefill", "Decode", "Mixed", "Saturated")
    found = {}
    for record in document.get("benchmarks", []):
        name = record.get("name", "")
        for scenario in expected:
            if name.startswith(f"{scenario}/1024/") and name.endswith("_p99"):
                found[scenario] = record
    missing = [scenario for scenario in expected if scenario not in found]
    if missing:
        raise SystemExit(f"error: missing 1024-request p99 records: {', '.join(missing)}")
    for scenario, record in found.items():
        latency_ns = nanoseconds(record["real_time"], record["time_unit"])
        if latency_ns > 1_000_000:
            raise SystemExit(f"error: {scenario} p99 {latency_ns} ns exceeds 1 ms")
        if record.get("allocations") != 0:
            raise SystemExit(f"error: {scenario} measured allocations are nonzero")
    print(
        "M1 scheduler gate:",
        " ".join(f"{name}={nanoseconds(row['real_time'], row['time_unit']):.0f}ns"
                 for name, row in found.items()),
        "allocations=0",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
