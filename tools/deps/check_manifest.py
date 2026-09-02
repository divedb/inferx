#!/usr/bin/env python3
"""Validate third_party/manifest.json against the repository's actual gitlinks.

The gitlink is the dependency lock; the manifest records intent (owner, license,
feature, status). This checker fails (nonzero exit) on any drift between the two,
between the manifest and `.gitmodules`, on schema violations, and on missing license
files for initialized dependencies. Uninitialized-but-declared submodules are reported
as notes, not errors, so the checker is useful on a fresh checkout.

Usage:
  tools/deps/check_manifest.py [--manifest PATH] [--root DIR] [--self-test]
"""

from __future__ import annotations

import argparse
import configparser
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED_STATUS = {"candidate", "approved", "qualified-deferred", "experimental", "rejected"}
REQUIRED_FIELDS = (
    "name",
    "source_kind",
    "path",
    "url",
    "revision",
    "license",
    "license_file",
    "owner",
    "features",
    "default_enabled",
    "status",
    "qualification",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class Drift(Exception):
    """A single drift or schema violation, reported with its path context."""


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def read_gitlinks(root: Path) -> dict[str, str]:
    """Map submodule path -> gitlink revision (works when worktrees are uninitialized)."""
    links: dict[str, str] = {}
    for line in run_git(root, "ls-files", "-s", "third_party").splitlines():
        mode, sha, _stage, path = line.split()
        if mode == "160000":
            links[path] = sha
    return links


def read_gitmodules(root: Path) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser()
    parser.read(root / ".gitmodules")
    modules: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        if not section.startswith("submodule "):
            continue
        path = parser.get(section, "path", fallback=None)
        if path:
            modules[path] = {
                "url": parser.get(section, "url", fallback=""),
                "branch": parser.get(section, "branch", fallback=""),
            }
    return modules


def validate_entry(entry: dict, index: int) -> None:
    where = f"dependencies[{index}]"
    if not isinstance(entry, dict):
        raise Drift(f"{where}: entry must be an object")
    name = entry.get("name", "<missing>")
    where = f"dependencies[{index}] ({name})"
    for field in REQUIRED_FIELDS:
        if field not in entry:
            raise Drift(f"{where}: missing required field '{field}'")
    if entry["source_kind"] not in ("git-submodule", "vendored-adaptation"):
        raise Drift(f"{where}: unsupported source_kind {entry['source_kind']!r}")
    if entry["source_kind"] == "vendored-adaptation":
        # In-tree qualified adaptation (ADR 0024): the revision records
        # upstream provenance; there is no gitlink to drift against, so only
        # the schema fields, status, and the license/qualification records
        # are enforced.
        if entry["status"] not in ALLOWED_STATUS:
            raise Drift(f"{where}: status {entry['status']!r} not in sorted({sorted(ALLOWED_STATUS)})")
        if not HEX40.match(entry["revision"]):
            raise Drift(f"{where}: revision {entry['revision']!r} is not a 40-hex sha1")
        if not entry["path"].startswith("third_party/"):
            raise Drift(f"{where}: path {entry['path']!r} must live under third_party/")
        if not entry["license_file"].startswith("third_party/"):
            raise Drift(f"{where}: license_file {entry['license_file']!r} must live under third_party/")
        if not isinstance(entry["features"], list) or not all(
            isinstance(f, str) for f in entry["features"]
        ):
            raise Drift(f"{where}: features must be a list of profile/feature names")
        if not isinstance(entry["default_enabled"], bool):
            raise Drift(f"{where}: default_enabled must be a boolean")
        return
    if entry["status"] not in ALLOWED_STATUS:
        raise Drift(f"{where}: status {entry['status']!r} not in sorted({sorted(ALLOWED_STATUS)})")
    if not HEX40.match(entry["revision"]):
        raise Drift(f"{where}: revision {entry['revision']!r} is not a 40-hex sha1")
    if not entry["path"].startswith("third_party/"):
        raise Drift(f"{where}: path {entry['path']!r} must live under third_party/")
    if not entry["license_file"].startswith("third_party/"):
        raise Drift(f"{where}: license_file {entry['license_file']!r} must live under third_party/")
    if not isinstance(entry["features"], list) or not all(
        isinstance(f, str) for f in entry["features"]
    ):
        raise Drift(f"{where}: features must be a list of profile/feature names")
    if not isinstance(entry["default_enabled"], bool):
        raise Drift(f"{where}: default_enabled must be a boolean")
    if not isinstance(entry.get("removed", False), bool):
        raise Drift(f"{where}: removed must be a boolean")
    if entry.get("removed") and entry["status"] != "rejected":
        raise Drift(f"{where}: removed=true requires status 'rejected'")


def check_manifest(manifest_path: Path, root: Path, *, notes: list[str] | None = None) -> list[str]:
    """Return a list of drift errors. Uninitialized dependencies append notes."""
    if notes is None:
        notes = []
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{manifest_path}: cannot read manifest: {exc}"]

    if manifest.get("schema_version") != 1:
        errors.append(f"{manifest_path}: schema_version must be 1")
    if not isinstance(manifest.get("dependencies"), list):
        return errors + [f"{manifest_path}: 'dependencies' must be a list"]

    try:
        gitlinks = read_gitlinks(root)
        gitmodules = read_gitmodules(root)
    except (RuntimeError, OSError) as exc:
        return errors + [f"cannot inspect repository state: {exc}"]

    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    seen_urls: set[str] = set()
    by_name: dict[str, dict] = {}

    for index, entry in enumerate(manifest["dependencies"]):
        try:
            validate_entry(entry, index)
        except Drift as drift:
            errors.append(str(drift))
            continue
        name, path = entry["name"], entry["path"]
        if name in seen_names:
            errors.append(f"dependencies[{index}] ({name}): duplicate name")
        if path in seen_paths:
            errors.append(f"dependencies[{index}] ({name}): duplicate path {path}")
        if entry["url"] in seen_urls:
            errors.append(f"dependencies[{index}] ({name}): duplicate url {entry['url']}")
        seen_names.add(name)
        seen_paths.add(path)
        seen_urls.add(entry["url"])
        by_name[name] = entry

        removed = entry.get("removed", False)
        has_gitlink = path in gitlinks
        in_gitmodules = path in gitmodules
        initialized = (root / path / ".git").exists()

        if removed:
            if has_gitlink:
                errors.append(
                    f"{path}: manifest marks the dependency removed but gitlink "
                    f"{gitlinks[path]} is still recorded; removal must delete the gitlink"
                )
            if in_gitmodules:
                errors.append(
                    f"{path}: manifest marks the dependency removed but .gitmodules "
                    "still declares it; removal must update .gitmodules"
                )
            continue

        if entry["source_kind"] == "vendored-adaptation":
            # In-tree adaptation: no gitlink by design; the tree's presence
            # and license file are the checks.
            if not (root / path).is_dir():
                errors.append(f"{path}: vendored adaptation directory is missing")
            if not (root / entry["license_file"]).exists():
                errors.append(
                    f"{path}: license file {entry['license_file']} is missing"
                )
            continue

        if not has_gitlink:
            errors.append(
                f"{path}: manifest declares the dependency but no gitlink exists "
                f"(manifest revision {entry['revision']})"
            )
            continue
        if gitlinks[path] != entry["revision"]:
            errors.append(
                f"{path}: revision drift: gitlink {gitlinks[path]} != manifest "
                f"{entry['revision']} (the gitlink is the lock; update both together)"
            )
        if not in_gitmodules:
            errors.append(f"{path}: gitlink exists but .gitmodules has no entry for it")
        elif gitmodules[path]["url"] != entry["url"]:
            errors.append(
                f"{path}: .gitmodules url {gitmodules[path]['url']!r} != manifest "
                f"url {entry['url']!r}"
            )
        if initialized and not (root / entry["license_file"]).is_file():
            errors.append(
                f"{entry['license_file']}: license file for initialized dependency "
                f"{name!r} is missing"
            )
        if not initialized:
            notes.append(
                f"{path}: uninitialized (gitlink {gitlinks[path][:12]}...); "
                "not required unless its feature is enabled"
            )

    profiles = manifest.get("profiles", {})
    if not isinstance(profiles, dict):
        errors.append("profiles: must be an object of profile name -> dependency names")
        profiles = {}
    profile_members: dict[str, set[str]] = {}
    for profile, members in profiles.items():
        if not isinstance(members, list):
            errors.append(f"profiles.{profile}: must be a list of dependency names")
            continue
        profile_members[profile] = set(members)
        for member in members:
            if member not in by_name:
                errors.append(f"profiles.{profile}: unknown dependency {member!r}")
            elif by_name[member].get("removed"):
                errors.append(f"profiles.{profile}: removed dependency {member!r} cannot be a member")

    for name, entry in by_name.items():
        if entry.get("removed"):
            continue
        if entry["default_enabled"] and entry["features"] and not any(
            name in profile_members.get(profile, set())
            for profile in entry["features"]
        ):
            errors.append(
                f"{entry['path']}: default_enabled dependency is not covered by any "
                f"of its feature profiles {entry['features']}"
            )
        if entry["status"] == "approved" and entry["features"] and not any(
            name in profile_members.get(profile, set()) for profile in entry["features"]
        ):
            errors.append(
                f"{entry['path']}: approved dependency is not initialized by any of "
                f"its feature profiles {entry['features']}"
            )

    extra_gitlinks = sorted(set(gitlinks) - seen_paths)
    for path in extra_gitlinks:
        errors.append(
            f"{path}: gitlink exists without a manifest entry; every source "
            "dependency must be declared"
        )
    return errors


def self_test() -> int:
    """Fixture-based self test: one valid repository plus mutated invalid variants."""
    with tempfile.TemporaryDirectory(prefix="inferx-manifest-selftest-") as tmp:
        root = Path(tmp)
        run = lambda *args: subprocess.run(  # noqa: E731 - local readability
            list(args), cwd=root, capture_output=True, text=True, check=True
        )
        run("git", "init", "-q", ".")
        run("git", "config", "user.email", "selftest@inferx.invalid")
        run("git", "config", "user.name", "manifest selftest")
        (root / ".gitmodules").write_text(
            '[submodule "third_party/alpha"]\n'
            "\tpath = third_party/alpha\n"
            "\turl = https://example.com/alpha.git\n",
            encoding="utf-8",
        )
        # A gitlink row in the index for an uninitialized submodule.
        (root / "third_party").mkdir()
        run("git", "update-index", "--add", "--cacheinfo", "160000,"
            "1111111111111111111111111111111111111111,third_party/alpha")
        valid_manifest = {
            "schema_version": 1,
            "profiles": {"core": ["alpha"]},
            "dependencies": [
                {
                    "name": "alpha",
                    "source_kind": "git-submodule",
                    "path": "third_party/alpha",
                    "url": "https://example.com/alpha.git",
                    "revision": "1" * 40,
                    "license": "MIT",
                    "license_file": "third_party/alpha/LICENSE",
                    "owner": "base",
                    "features": ["core"],
                    "default_enabled": True,
                    "status": "approved",
                    "qualification": "docs/dependencies/alpha.md",
                }
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(valid_manifest), encoding="utf-8")
        failures: list[str] = []

        def expect(outcome: str, errors: list[str], label: str) -> None:
            ok = (not errors) if outcome == "valid" else bool(errors)
            if not ok:
                failures.append(f"{label}: expected {outcome}, got errors={errors}")

        expect("valid", check_manifest(manifest_path, root), "valid fixture")

        def mutate(label: str, **changes) -> None:
            entry = dict(valid_manifest["dependencies"][0], **changes)
            mutated = dict(valid_manifest, dependencies=[entry])
            manifest_path.write_text(json.dumps(mutated), encoding="utf-8")
            expect("invalid", check_manifest(manifest_path, root), label)

        mutate("bad status", status="garbage")
        mutate("bad revision", revision="deadbeef")
        mutate("path outside third_party", path="vendor/alpha")
        mutate("license outside third_party", license_file="LICENSE")
        mutate("removed without rejected", removed=True)

        # Removed entries must actually be gone from git, .gitmodules, and profiles.
        run("git", "update-index", "--force-remove", "third_party/alpha")
        (root / ".gitmodules").write_text("", encoding="utf-8")
        removed_entry = dict(valid_manifest["dependencies"][0], status="rejected", removed=True,
                             features=[], default_enabled=False)
        manifest_path.write_text(
            json.dumps(dict(valid_manifest, dependencies=[removed_entry], profiles={})),
            encoding="utf-8",
        )
        expect("valid", check_manifest(manifest_path, root), "removed fixture")

        # A gitlink with no manifest entry is drift.
        run("git", "update-index", "--add", "--cacheinfo", "160000,"
            "2222222222222222222222222222222222222222,third_party/beta")
        errors = check_manifest(manifest_path, root)
        if not any("third_party/beta" in e for e in errors):
            failures.append(f"undeclared gitlink not detected: {errors}")

        if failures:
            for failure in failures:
                print(f"self-test FAIL: {failure}", file=sys.stderr)
            return 1
        print("check_manifest self-test: all fixtures behaved as expected")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--manifest", type=Path,
                        default=default_root / "third_party" / "manifest.json")
    parser.add_argument("--root", type=Path, default=default_root,
                        help="repository root used to read gitlinks and .gitmodules")
    parser.add_argument("--self-test", action="store_true",
                        help="run built-in valid/invalid fixtures and exit")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    notes: list[str] = []
    errors = check_manifest(args.manifest.resolve(), args.root.resolve(), notes=notes)
    for note in notes:
        print(f"note: {note}")
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(
            "dependency manifest drift detected; fix the manifest, gitlinks, and "
            ".gitmodules together (see docs/dependency-policy.md)",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {args.manifest} agrees with gitlinks and .gitmodules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
