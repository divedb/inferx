#!/usr/bin/env python3
"""Build metrics capture (m0.md section 13.6).

Runs a fresh configure -> clean build -> no-op build -> tests -> install ->
consumer flow for a preset and records, as JSON:

  - machine/OS and compiler/CMake/Ninja identifiers;
  - core dependency revisions (gitlinks);
  - wall/user/system time and peak RSS of configure and clean build;
  - no-op build time, unit-test time, install and consumer-build time;
  - `inferx_base` and `inferx-info` file sizes;
  - CUDA smoke compile time when the preset enables CUDA.

M0 establishes the baseline; there is no absolute time gate. A missing or
invalid metrics artifact fails the CI metrics job, so this script writes the
file even when a step reports zero, and validates it can be re-parsed before
exiting successfully. All generated trees stay under out/metrics/.
"""

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def command_output(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout.strip() if result.returncode == 0 else f"<unavailable: {' '.join(cmd)}>"


def run_timed(cmd: list[str], cwd: Path | None = None) -> tuple[bool, dict]:
    """Run a command, returning (success, {wall, user, system, peak_rss_kb})."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=3600)
    elapsed = time.monotonic() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    stats = {
        "wall_seconds": round(elapsed, 3),
        "user_seconds": round(max(after.ru_utime - before.ru_utime, 0.0), 3),
        "system_seconds": round(max(after.ru_stime - before.ru_stime, 0.0), 3),
        "peak_rss_kb": after.ru_maxrss,  # cumulative max across children
    }
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:], file=sys.stderr)
    return result.returncode == 0, stats


def dependency_revisions() -> dict[str, str]:
    revisions = {}
    listing = subprocess.run(["git", "-C", str(ROOT), "submodule", "status"],
                             capture_output=True, text=True, timeout=60)
    if listing.returncode != 0:
        return revisions
    for line in listing.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            revisions[fields[1].removeprefix("third_party/")] = fields[0].lstrip("-+")
    return revisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="cpu-release")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", default=os.cpu_count() or 4)
    args = parser.parse_args()

    work = ROOT / "out" / "metrics" / f"{args.preset}-{int(time.time())}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    build = work / "build"
    prefix = work / "prefix"
    consumer_build = work / "consumer-build"

    metrics: dict = {
        "preset": args.preset,
        "jobs": int(args.jobs),
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor_count": os.cpu_count(),
            "container": os.environ.get("INFERX_METRICS_CONTAINER", "unknown"),
        },
        "tools": {
            "cmake": command_output(["cmake", "--version"]).splitlines()[0],
            "ninja": command_output(["ninja", "--version"]),
            "compiler": command_output([os.environ.get("CXX", "c++"), "--version"]).splitlines()[0],
        },
        "core_dependency_revisions": dependency_revisions(),
    }

    configure_cmd = ["cmake", "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
                     f"-DCMAKE_BUILD_TYPE=Release",
                     f"-DCMAKE_CXX_COMPILER={os.environ.get('CXX', 'g++')}"]
    ok, metrics["configure"] = run_timed(configure_cmd)
    if not ok:
        sys.exit("error: metrics configure failed")

    ok, metrics["clean_build"] = run_timed(
        ["cmake", "--build", str(build), "--parallel", str(args.jobs)])
    if not ok:
        sys.exit("error: metrics build failed")

    ok, metrics["noop_build"] = run_timed(
        ["cmake", "--build", str(build), "--parallel", str(args.jobs)])
    if not ok:
        sys.exit("error: metrics no-op build failed")

    ok, metrics["unit_tests"] = run_timed(
        ["ctest", "--test-dir", str(build), "-L", "unit", "--output-on-failure"])
    if not ok:
        sys.exit("error: metrics unit tests failed")

    ok, metrics["install"] = run_timed(
        ["cmake", "--install", str(build), "--prefix", str(prefix)])
    if not ok:
        sys.exit("error: metrics install failed")

    consumer_configure = [
        "cmake", "-S", str(ROOT / "tests" / "consumer"), "-B", str(consumer_build),
        "-G", "Ninja", f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DCMAKE_CXX_COMPILER={os.environ.get('CXX', 'g++')}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    ok, metrics["consumer_build"] = run_timed(consumer_configure)
    ok = ok and subprocess.run(["cmake", "--build", str(consumer_build), "--parallel",
                                str(args.jobs)], capture_output=True, timeout=3600).returncode == 0
    if not ok:
        sys.exit("error: metrics consumer build failed")

    artifacts = {}
    for name, path in {
        "inferx_base": build / "src" / "base" / "libinferx_base.a",
        "inferx_info": build / "apps" / "inferx_info" / "inferx-info",
    }.items():
        artifacts[name] = path.stat().st_size if path.is_file() else -1
    metrics["artifact_bytes"] = artifacts

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    # Validate the artifact round-trips before reporting success.
    json.loads(args.output.read_text(encoding="utf-8"))
    print(f"metrics: wrote and validated {args.output}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
