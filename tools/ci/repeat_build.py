#!/usr/bin/env python3
"""Repeated fresh-build stress driver.

Creates distinct, validated paths below out/stress/ and repeats, at least
three times by default:

    fresh configure -> parallel build -> test -> install -> consumer test

It runs once with parallelism 1, once with the CI allocation (or --jobs), and
once with tests discovered in a different order (--shuffle). It never depends
on a previous generated file, never writes to the source tree, and refuses
dangerous output paths ('/', a home directory, the repository root, or
anything outside out/stress/ or an explicit validated temporary root).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def validate_output_root(candidate: Path) -> Path:
    resolved = candidate.resolve()
    allowed = [
        (ROOT / "out" / "stress").resolve(),
        Path(tempfile.gettempdir()),
    ]
    dangerous = {Path("/"), Path.home().resolve(), ROOT.resolve()}
    if resolved in dangerous:
        sys.exit(f"error: refusing dangerous output root {resolved}")
    if not any(resolved == base or str(resolved).startswith(str(base) + os.sep)
               for base in allowed):
        sys.exit(f"error: output root must live under out/stress/ or the system "
                 f"temp directory, not {resolved}")
    if resolved.exists() and not resolved.is_dir():
        sys.exit(f"error: output root {resolved} exists and is not a directory")
    return resolved


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    printable = " ".join(str(part) for part in cmd)
    print(f"+ {printable}", flush=True)
    result = subprocess.run([str(part) for part in cmd], capture_output=True,
                            text=True, timeout=3600, **kwargs)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)
        sys.exit(f"error: stress iteration failed: {printable}")
    return result


def iteration(name: str, output_root: Path, jobs: int, *, shuffle: bool) -> None:
    work = output_root / f"{name}-{int(time.time())}"
    work.mkdir(parents=True)
    build = work / "build"
    prefix = work / "prefix"
    consumer = work / "consumer-build"
    compiler = os.environ.get("CXX", "g++")

    run(["cmake", "-S", ROOT, "-B", build, "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CXX_COMPILER={compiler}"])
    run(["cmake", "--build", build, "--parallel", str(jobs)])
    test_cmd = ["ctest", "--test-dir", build, "--output-on-failure"]
    if shuffle:
        test_cmd.append("--shuffle")
    run(test_cmd)
    run(["cmake", "--install", build, "--prefix", prefix])
    run(["cmake", "-S", ROOT / "tests" / "consumer", "-B", consumer, "-G", "Ninja",
         f"-DCMAKE_PREFIX_PATH={prefix}", f"-DCMAKE_CXX_COMPILER={compiler}",
         "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", consumer, "--parallel", str(jobs)])
    run([consumer / "inferx_consumer"],
        env={**os.environ, "INFERX_EXPECTED_VERSION": "0.0.0"})
    shutil.rmtree(work, ignore_errors=False)
    print(f"stress: iteration {name} complete (jobs={jobs}, shuffle={shuffle})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "out" / "stress")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4,
                        help="CI allocation parallelism for the fast lane")
    parser.add_argument("--repeats", type=int, default=3,
                        help="minimum repeats of the full flow (default 3)")
    args = parser.parse_args()

    output_root = validate_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Lane 1: single-job build proves no hidden parallel-order dependency.
    iteration("serial", output_root, jobs=1, shuffle=False)
    # Lane 2: CI allocation with shuffled test discovery order.
    iteration("parallel-shuffled", output_root, jobs=args.jobs, shuffle=True)
    # Lanes 3+: plain repeats at the CI allocation.
    for index in range(args.repeats - 2):
        iteration(f"repeat-{index}", output_root, jobs=args.jobs, shuffle=False)

    print(f"stress: {args.repeats} repeated fresh builds passed under {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
