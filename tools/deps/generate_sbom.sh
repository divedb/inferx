#!/usr/bin/env bash
# SPDX SBOM generation for an installed InferX prefix.
#
# Usage: tools/deps/generate_sbom.sh <install-prefix> <output.spdx.json>
#
# Uses a pinned Syft release (version + sha256 below) so generation is
# reproducible. Resolution order: SYFT_BIN env override, a matching `syft` on
# PATH, a pinned download in the user cache, then a pinned docker image. The
# script writes only below the output path and the user cache, records the
# Syft version/digest in a sidecar, validates the JSON before returning
# success, and never uploads anything.
set -euo pipefail

readonly SYFT_VERSION="v1.51.1"
readonly SYFT_SHA256="8fcb33017a0dc1058298c923c436d19dfa68ae93968e0b423248542e3afb9fc3"
readonly SYFT_URL="https://github.com/anchore/syft/releases/download/${SYFT_VERSION}/syft_1.51.1_linux_amd64.tar.gz"
readonly SYFT_IMAGE="anchore/syft:${SYFT_VERSION}"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <install-prefix> <output.spdx.json>" >&2
  exit 2
fi

PREFIX="$(cd "$1" && pwd)"
OUTPUT="$2"
mkdir -p "$(dirname "$OUTPUT")"
OUTPUT="$(cd "$(dirname "$OUTPUT")" && pwd)/$(basename "$OUTPUT")"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/inferx/syft/${SYFT_VERSION}"

find_syft() {
  if [[ -n "${SYFT_BIN:-}" ]]; then
    echo "${SYFT_BIN}"
    return 0
  fi
  if command -v syft >/dev/null 2>&1; then
    if syft version 2>/dev/null | grep -q "Version:.*${SYFT_VERSION#v}"; then
      echo syft
      return 0
    fi
    echo "note: PATH syft is not the pinned ${SYFT_VERSION}; using the pinned binary" >&2
  fi
  local cached="${CACHE_DIR}/syft"
  if [[ ! -x "${cached}" ]]; then
    mkdir -p "${CACHE_DIR}"
    local tgz="${CACHE_DIR}/syft.tar.gz"
    echo "downloading pinned syft ${SYFT_VERSION} to ${CACHE_DIR}" >&2
    curl -fsSL -o "${tgz}" "${SYFT_URL}"
    echo "${SYFT_SHA256}  ${tgz}" | sha256sum -c - >/dev/null
    tar -xzf "${tgz}" -C "${CACHE_DIR}" syft
    rm -f "${tgz}"
  fi
  echo "${cached}"
}

run_syft() {
  local bin
  if ! bin="$(find_syft)"; then
    return 1
  fi
  "${bin}" scan "dir:${PREFIX}" --source-name inferx-install \
    --source-version "${INFERX_SBOM_VERSION:-0.0.0}" -o "spdx-json=${OUTPUT}"
}

if ! run_syft; then
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "falling back to pinned docker image ${SYFT_IMAGE}" >&2
    docker run --rm --network=none -v "${PREFIX}:/prefix:ro" -v "$(dirname "${OUTPUT}"):/out" \
      "${SYFT_IMAGE}" scan "dir:/prefix" -o "spdx-json=/out/$(basename "${OUTPUT}")"
  else
    echo "error: cannot obtain the pinned syft ${SYFT_VERSION}: download failed and no docker daemon" >&2
    exit 1
  fi
fi

# Merge the source-dependency manifest into the SPDX package list so the SBOM
# records the pinned gitlink provenance alongside the installed artifacts.
python3 - "$OUTPUT" "$ROOT/third_party/manifest.json" <<'EOF'
import json
import sys

output_path, manifest_path = sys.argv[1], sys.argv[2]
sbom = json.load(open(output_path, encoding="utf-8"))
manifest = json.load(open(manifest_path, encoding="utf-8"))
sbom.setdefault("packages", []).append({
    "name": "inferx-third-party-manifest",
    "SPDXID": "SPDXRef-Package-InferXManifest",
    "versionInfo": f"schema-{manifest.get('schema_version', 1)}",
    "downloadLocation": "NOASSERTION",
    "licenseConcluded": "NOASSERTION",
    "licenseDeclared": "NOASSERTION",
    "copyrightText": "NOASSERTION",
    "description": "Source dependencies: "
    + "; ".join(
        f"{d['name']}@{d['revision'][:12]} status={d['status']}"
        f" removed={'removed' in d}"
        for d in manifest.get("dependencies", [])
    ),
})
json.dump(sbom, open(output_path, "w", encoding="utf-8"), indent=2)
EOF

# Validate JSON before reporting success.
python3 - "${OUTPUT}" <<'EOF'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    document = json.load(handle)
assert document.get("SPDXID") == "SPDXRef-DOCUMENT", "not an SPDX document"
assert document.get("spdxVersion", "").startswith("SPDX-2"), "unexpected SPDX version"
print(f"sbom: {sys.argv[1]} is valid SPDX JSON "
      f"({len(document.get('packages', []))} packages)")
EOF

printf 'syft %s (sha256 %s of the linux-amd64 tarball)\n' \
  "${SYFT_VERSION}" "${SYFT_SHA256}" > "${OUTPUT}.syft.txt"
echo "sbom: recorded generator in ${OUTPUT}.syft.txt; nothing was uploaded"
