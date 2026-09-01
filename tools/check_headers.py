#!/usr/bin/env python3
"""Public header self-containment probes (m0.md section 9.3).

Generates one temporary translation unit per installed public header
containing only that include and compiles it with the supported compiler and
the same language requirement a consumer uses (C++23, extensions off). This
catches transitive-include reliance independently of IWYU.

Probes live under out/ (or a validated temporary directory), never enter Git,
and never write to the source tree.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def compiler_from_build_dir(build_dir: Path) -> str | None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
            return line.split("=", 1)[1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=None,
                        help="build tree used to locate the configured compiler")
    parser.add_argument("--compiler", default=None,
                        help="compiler override (default: build dir, then c++/g++/clang++)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="probe directory (default: a temporary directory)")
    args = parser.parse_args()

    include_root = args.source_root / "include"
    headers = sorted(include_root.glob("inferx/**/*.h"))
    if not headers:
        sys.exit(f"error: no public headers found under {include_root}")

    compiler = args.compiler
    if compiler is None and args.build_dir is not None:
        compiler = compiler_from_build_dir(args.build_dir)
    if compiler is None:
        for candidate in ("c++", "g++", "clang++"):
            probe = subprocess.run(["sh", "-c", f"command -v {candidate}"],
                                   capture_output=True, text=True)
            if probe.returncode == 0:
                compiler = candidate
                break
    if compiler is None:
        sys.exit("error: no C++ compiler found; pass --compiler")

    work_dir = args.work_dir
    cleanup = None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="inferx-header-probes-"))
        cleanup = work_dir
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        resolved = work_dir.resolve()
        allowed_roots = [(args.source_root / "out").resolve(), Path(tempfile.gettempdir())]
        if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
            sys.exit(f"error: probe directory must live under out/ or the system "
                     f"temp directory, not {resolved}")

    failures: list[str] = []
    # Since M1 (ADR 0008), public headers include Abseil headers. A consumer
    # sees them beside inferx/ in the installed prefix's include directory;
    # for the in-tree probe the pinned Abseil source provides the same
    # absl/... layout.
    extra_includes = []
    absl_root = args.source_root / "third_party" / "abseil-cpp"
    if absl_root.is_dir():
        extra_includes = ["-I", str(absl_root)]
    for header in headers:
        probe = work_dir / (str(header.relative_to(include_root)).replace("/", "_") + ".cc")
        probe.write_text(f"#include <{header.relative_to(include_root)}>\n\n"
                         "int main() { return 0; }\n", encoding="utf-8")
        result = subprocess.run(
            [compiler, "-std=c++23", "-I", str(include_root), *extra_includes,
             "-Wall", "-Wextra", "-Werror", "-c", str(probe), "-o", str(probe) + ".o"],
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            failures.append(header.name)
            print(f"header probe FAILED: {header.relative_to(include_root)}")
            print(result.stderr[:2000])

    if cleanup is not None:
        import shutil
        shutil.rmtree(cleanup, ignore_errors=True)

    if failures:
        sys.exit(f"error: {len(failures)} public header(s) are not self-contained: "
                 f"{', '.join(failures)}")
    print(f"header probes: {len(headers)} public header(s) self-contained "
          f"({compiler}, C++23)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
