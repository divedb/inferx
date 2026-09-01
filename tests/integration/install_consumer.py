#!/usr/bin/env python3
"""Install + independent consumer integration driver (m0.md section 6.7).

For tests/consumer: installs InferX into a prefix and configures a separate
CMake project that locates it only through CMAKE_PREFIX_PATH. For
tests/consumer_absl: first installs the pinned Abseil into its own prefix with
its install rules enabled (the dedicated installed-dependency fixture), then
consumes it the same way.

Logs display the install prefix and the exact consumer configure command so
evidence shows no source-tree or build-tree leakage.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    printable = " ".join(cmd)
    print(f"+ {printable}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(f"error: command failed ({result.returncode}): {printable}")
    return result


def consumer_kind(consumer_dir: Path) -> str:
    return "absl" if consumer_dir.name == "consumer_absl" else "inferx"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True,
                        help="configured InferX build tree to install from")
    parser.add_argument("--consumer-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="all generated trees are created below this directory")
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    kind = consumer_kind(args.consumer_dir)
    stamp = f"{int(time.time())}-{os.getpid()}"
    work = args.output_root / f"{kind}-{stamp}"
    work.mkdir(parents=True, exist_ok=True)

    # The pinned Abseil is installed into the prefix for both fixtures: the
    # absl-only one proves the installed-dependency strategy, and since M1
    # (ADR 0008) inferx::base exposes Abseil in public headers so the InferX
    # consumer needs the package in the same prefix too.
    prefix = work / ("absl-prefix" if kind == "absl" else "prefix")
    print(f"[install-consumer] install prefix: {prefix}")
    absl_build = work / "absl-build"
    run(["cmake", "-S", str(args.source_dir / "third_party" / "abseil-cpp"),
         "-B", str(absl_build), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CXX_COMPILER=" + args.cxx_compiler,
         "-DCMAKE_CXX_STANDARD=23",
         "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
         "-DCMAKE_CXX_EXTENSIONS=OFF",
         "-DCMAKE_INSTALL_PREFIX=" + str(prefix),
         "-DABSL_PROPAGATE_CXX_STD=ON",
         "-DABSL_ENABLE_INSTALL=ON",
         "-DBUILD_TESTING=OFF"])
    run(["cmake", "--build", str(absl_build), "--parallel"])
    run(["cmake", "--install", str(absl_build)])

    if kind == "inferx":
        run(["cmake", "--install", str(args.build_dir), "--prefix", str(prefix)])

    consumer_build = work / "consumer-build"
    configure_cmd = [
        "cmake", "-S", str(args.consumer_dir), "-B", str(consumer_build), "-G", "Ninja",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DCMAKE_CXX_COMPILER={args.cxx_compiler}",
        f"-DCMAKE_BUILD_TYPE=Release",
    ]
    print(f"[install-consumer] independent consumer configure: {' '.join(configure_cmd)}")
    run(configure_cmd)
    run(["cmake", "--build", str(consumer_build), "--parallel"])

    binary = consumer_build / ("inferx_absl_consumer" if kind == "absl" else "inferx_consumer")
    result = run([str(binary)], env={**os.environ, "INFERX_EXPECTED_VERSION": args.expected_version})
    if "consumer ok" not in result.stdout:
        sys.exit(f"error: consumer binary did not validate: {result.stdout!r}")
    print(f"[install-consumer] {kind} consumer ok")

    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
