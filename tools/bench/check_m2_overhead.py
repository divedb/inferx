#!/usr/bin/env python3
"""Calculate M2 benchmark percentiles and enforce paired wrapper overhead."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def normalized_name(name: str) -> str:
    return "/".join(part for part in name.split("/") if not part.startswith("repeats:"))


def load_samples(path: Path) -> dict[str, list[float]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    samples: dict[str, list[float]] = {}
    for entry in document.get("benchmarks", []):
        if entry.get("run_type", "iteration") != "iteration":
            continue
        name = normalized_name(entry["name"])
        samples.setdefault(name, []).append(float(entry["real_time"]))
    return samples


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples: dict[str, list[float]] = {}
    for path in (args.copy, args.event, args.launch):
        samples.update(load_samples(path))
    metrics = {
        name: {
            "samples": len(values),
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p99": percentile(values, 0.99),
        }
        for name, values in sorted(samples.items())
        if values
    }

    name_pairs = [
        ("BM_CudaPinnedH2DWrapper", "BM_CudaPinnedH2DDirect"),
        ("BM_CudaPinnedD2HWrapper", "BM_CudaPinnedD2HDirect"),
        ("BM_CudaD2DWrapper", "BM_CudaD2DDirect"),
        ("BM_CudaEventRecordWaitAcknowledge", "BM_CudaEventRecordQueryDirect"),
        ("BM_CudaNoOpLaunchWrapper", "BM_CudaNoOpLaunchDirect"),
    ]
    comparisons = []
    failed = False
    for wrapper_prefix, direct_prefix in name_pairs:
        wrapper_names = [name for name in metrics if name.startswith(wrapper_prefix)]
        for wrapper_name in wrapper_names:
            suffix = wrapper_name[len(wrapper_prefix) :]
            direct_name = direct_prefix + suffix
            if direct_name not in metrics:
                print(f"missing direct benchmark for {wrapper_name}", file=sys.stderr)
                failed = True
                continue
            wrapper_p50 = metrics[wrapper_name]["p50"]
            direct_p50 = metrics[direct_name]["p50"]
            overhead = (wrapper_p50 / direct_p50) - 1.0
            passed = overhead <= 0.03
            comparisons.append(
                {
                    "wrapper": wrapper_name,
                    "direct": direct_name,
                    "overhead_fraction": overhead,
                    "limit_fraction": 0.03,
                    "passed": passed,
                }
            )
            failed = failed or not passed
    if len(comparisons) < 5:
        print("M2 overhead evidence is incomplete", file=sys.stderr)
        failed = True

    result = {
        "schema_version": 1,
        "metrics": metrics,
        "comparisons": comparisons,
        "passed": not failed,
    }
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    if failed:
        print("M2 wrapper overhead exceeds 3% or evidence is incomplete", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
