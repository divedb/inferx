# Internal use only

Until [ADR 0007](docs/adr/0007-project-licensing.md) is accepted by the repository
owner, this repository and anything built from it are **internal use only**:

- no source or binary distribution outside the owner's control;
- no public package or container publication; and
- no representation of licensing terms to third parties.

Dependency licenses and notice obligations are tracked per dependency in
[`third_party/manifest.json`](third_party/manifest.json) and
[`docs/dependencies/`](docs/dependencies/). The SBOM procedure
(`tools/deps/generate_sbom.sh`) inventories what a future distribution decision would
need to clear.
