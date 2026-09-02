#!/usr/bin/env python3
"""Initialize pinned source dependencies from third_party/manifest.json.

This is the only documented initializer (ADR 0003). Profiles initialize only the
direct dependencies a feature needs; CMake itself never touches the network. After
bootstrap, configure/build/test run fully offline.

Usage:
  tools/deps/bootstrap.py --profile core        # Abseil, simdjson, BLAKE3, tests/benchmark
  tools/deps/bootstrap.py --list
  tools/deps/bootstrap.py --dependency cutlass  # initialize a single deferred dep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "third_party" / "manifest.json"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, **kwargs)


def load_manifest() -> dict:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read {MANIFEST}: {exc}")
    if manifest.get("schema_version") != 1:
        sys.exit(f"error: unsupported manifest schema_version {manifest.get('schema_version')!r}")
    return manifest


def select(manifest: dict, profile: str | None, dependency: str | None) -> list[dict]:
    deps = {entry["name"]: entry for entry in manifest["dependencies"]}
    if dependency is not None:
        if dependency not in deps:
            sys.exit(f"error: unknown dependency {dependency!r}; known: {sorted(deps)}")
        return [deps[dependency]]
    if profile not in manifest.get("profiles", {}):
        known = ", ".join(sorted(manifest.get("profiles", {}))) or "(none)"
        sys.exit(f"error: unknown profile {profile!r}; known profiles: {known}")
    # Only git-submodule entries are initialized here; vendored trees (e.g.
    # the tokenizer package) live in-tree and need no bootstrap.
    return [deps[name] for name in manifest["profiles"][profile]
            if deps[name].get("source_kind") == "git-submodule"]


def bootstrap(entry: dict) -> bool:
    path = ROOT / entry["path"]
    print(f"==> {entry['name']} @ {entry['revision'][:12]} ({path.relative_to(ROOT)})")
    result = run(
        ["git", "submodule", "update", "--init", "--depth", "1", entry["path"]],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"error: failed to initialize {entry['path']}", file=sys.stderr)
        return False
    checked_out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if checked_out != entry["revision"]:
        print(
            f"error: {entry['path']} is at {checked_out} but the manifest pins "
            f"{entry['revision']}; resolve the drift with "
            f"python3 tools/deps/check_manifest.py (the gitlink is the lock)",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile", help="profile name from third_party/manifest.json")
    group.add_argument("--dependency", help="initialize a single named dependency")
    group.add_argument("--list", action="store_true", help="list profiles and statuses")
    args = parser.parse_args()

    manifest = load_manifest()
    if args.list:
        for profile, members in sorted(manifest.get("profiles", {}).items()):
            print(f"{profile}: {' '.join(members)}")
        for entry in manifest["dependencies"]:
            state = "removed" if entry.get("removed") else entry["status"]
            initialized = (ROOT / entry["path"] / ".git").exists()
            print(f"  {entry['name']:<12} {state:<20} "
                  f"{'initialized' if initialized else 'not initialized'}")
        return 0

    entries = select(manifest, args.profile, args.dependency)
    if not entries:
        print("nothing to initialize", file=sys.stderr)
        return 0
    failed = [entry["name"] for entry in entries if not bootstrap(entry)]
    if failed:
        print(f"bootstrap failed for: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("bootstrap complete; configure/build/test now run offline", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
