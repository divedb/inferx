#!/usr/bin/env python3
"""Manifest/gitlink drift fixture (m0.md 13.5 row 10).

Copies the real manifest, mutates the mirrored abseil-cpp revision to a
different 40-hex value, and runs the checker against the repository's actual
gitlinks. The checker must exit nonzero naming the path and both revisions —
the gitlink is the lock.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "third_party" / "manifest.json"
    checker = root / "tools" / "deps" / "check_manifest.py"

    with tempfile.TemporaryDirectory(prefix="inferx-drift-") as tmp:
        mutated = Path(tmp) / "manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in data["dependencies"]:
            if entry["name"] == "abseil-cpp":
                # Distinct but still a plausible 40-hex sha so the schema
                # checks pass and only the drift check can fire.
                entry["revision"] = "f" * 40
        mutated.write_text(json.dumps(data), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(checker),
             "--manifest", str(mutated), "--root", str(root)],
            capture_output=True, text=True, timeout=60)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
