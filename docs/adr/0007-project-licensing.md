# ADR 0007: Project licensing and distribution terms

- Status: Proposed — **blocked on repository owner decision**; no coding agent may
  select distribution terms
- Date: 2026-08-30
- Deciding owner: repository owner (with legal review where applicable)
- Seed: `docs/milestones/m0.md` sections 2, 4; `INTERNAL_USE_ONLY.md`

## Context

The repository has no root license or distribution notice. Distribution terms are a
business/legal decision: they bind contributors, downstream consumers, and any future
public release, and they interact with dependency licenses (Apache-2.0 Abseil,
GoogleTest, Google Benchmark; BSL-1.0 Boost; MIT/BSD variants in the deferred
submodules). Choosing a license to "unblock" a build would misrepresent ownership.

## Decision

1. **Default posture until accepted:** the project is internal-use only. No binary or
   source distribution outside the owner's control. `INTERNAL_USE_ONLY.md` states this
   at the repository root and is installed with the package metadata.
2. **Required before any distribution:** the repository owner records the chosen
   license in this ADR (`Accepted`), adds the corresponding `LICENSE` text at the root,
   updates `INTERNAL_USE_ONLY.md`/install rules accordingly, and re-runs the SBOM +
   license-validation workflow against the dependency closure of everything actually
   shipped.
3. **Dependency rule regardless of outcome:** no dependency may be promoted above
   `experimental` in `third_party/manifest.json` without its license recorded and its
   notice obligations documented in `docs/dependencies/<name>.md`.

## Alternatives

- **Pick a permissive license now (MIT/Apache-2.0):** rejected; not the agent's or any
  engineer's unilateral call.
- **Proprietary/closed notice now:** likewise an owner decision, not a default.
- **No notice at all:** rejected; ambiguous terms are worse than an explicit
  internal-use statement.

## Consequences

- M0 can complete its engineering gates with this ADR in `Proposed` state; **public
  binary distribution is blocked** until it is `Accepted`.
- CI license validation covers initialized approved/candidate dependencies only; the
  SBOM records what a future distribution would need to clear.

## Validation evidence

- `INTERNAL_USE_ONLY.md` present and installed under `${CMAKE_INSTALL_DOCDIR}`.
- `tools/deps/check_manifest.py` verifies each initialized approved/candidate
  dependency exposes its declared license file (CI `policy` job).
- SBOM smoke artifact (`sbom` CI job) inventories the CPU install prefix.

## Supersession

This ADR supersedes nothing and is superseded by the owner's accepted licensing
decision recorded here (state changes to `Accepted`, license text added in the same
change).
