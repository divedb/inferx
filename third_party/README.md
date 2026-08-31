# third_party

Pinned source dependencies. The **gitlinks are the lock**;
[`manifest.json`](manifest.json) records intent (owner, license, feature, status) and is
drift-checked against the gitlinks and `.gitmodules` by
`python3 tools/deps/check_manifest.py`. Policy: [`docs/dependency-policy.md`](../docs/dependency-policy.md);
per-dependency evidence: [`docs/dependencies/`](../docs/dependencies/).

## Initialization

```bash
python3 tools/deps/bootstrap.py --profile core   # Abseil, GoogleTest, Google Benchmark
python3 tools/deps/bootstrap.py --list           # profiles and initialization state
python3 tools/deps/bootstrap.py --dependency cutlass  # a single deferred dependency
```

CMake never touches the network; after bootstrap, configure/build/test run offline.

## Current dependencies

| Name | Status | Profile/feature | License | Report |
|---|---|---|---|---|
| abseil-cpp | approved | core | Apache-2.0 | [report](../docs/dependencies/abseil-cpp.md) |
| googletest | approved (test-only) | core | BSD-3-Clause | [report](../docs/dependencies/googletest.md) |
| benchmark | approved (test-only) | core | Apache-2.0 | [report](../docs/dependencies/benchmark.md) |
| beast | qualified-deferred (M9) | http-server | BSL-1.0 | [report](../docs/dependencies/beast.md) |
| cutlass | qualified-deferred (M4) | kernels | BSD-3-Clause | [report](../docs/dependencies/cutlass.md) |
| flashinfer | candidate (M4) | kernels | Apache-2.0 | [report](../docs/dependencies/flashinfer.md) |
| tokenizer | candidate (M3) | tokenization | MIT | [report](../docs/dependencies/tokenizer.md) |
| hpc-ops | experimental, disabled | experimental-kernels | MIT + exceptions | [report](../docs/dependencies/hpc-ops.md) |
| folly | rejected (removed) | — | Apache-2.0 (historical) | [report](../docs/dependencies/folly.md) |

## Generated evidence

- License inventory: `manifest.json` (validated in CI) plus per-dependency reports.
- SBOM: `tools/deps/generate_sbom.sh <install-prefix> <output.spdx.json>` produces SPDX
  JSON from the installed prefix; artifacts are uploaded by CI, never committed.
