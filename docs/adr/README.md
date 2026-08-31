# Architecture decision records

This directory records decisions that other code and documents depend on. An ADR is
numbered, titled after its decision, and never edited in place once accepted — a change
supersedes the old record with a new one.

## States

| State | Meaning |
|---|---|
| `Proposed` | Written and reasoned, not yet binding. Implementation may proceed provisionally where the ADR says so, but dependent gates cannot close until acceptance. |
| `Accepted` | Binding until superseded. The decision, evidence, and links are complete. |
| `Superseded` | Replaced by a later ADR; the header links to the successor. Kept for history. |
| `Rejected` | Considered and declined; records why so it is not re-proposed without new evidence. |

## Required sections

Every ADR contains, in order:

1. **Status** — one of the states above, plus date and the deciding owner/role.
2. **Context** — the forces and constraints that made the decision necessary.
3. **Decision** — the active sentence(s), specific enough to test a change against.
4. **Alternatives** — the serious options considered and why each lost.
5. **Consequences** — what becomes easier, harder, or newly required.
6. **Validation evidence** — commands, CI jobs, test names, or reports that demonstrate the decision holds. Updated when the evidence moves.
7. **Supersession** — how this record may be replaced (link by number).

## Rules

- Do not park unresolved choices in comments inside CMake files or source; write the ADR.
- A change to an accepted decision requires a new ADR that links the old one.
- `docs/milestones/m0.md` section 4 is the seed list for ADR 0001–0007; later milestones
  add their own records.
- Cross-link ADRs from the implementation points they govern (for example the dependency
  module links ADR 0003, the sanitizer module links ADR 0006 lanes).

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-initial-product-scope.md) | Initial product scope | Proposed |
| [0002](0002-error-contract.md) | Cross-module error contract | Proposed |
| [0003](0003-dependency-management.md) | Dependency management and offline builds | Proposed |
| [0004](0004-exception-and-abi-boundaries.md) | Exception, RTTI, visibility, and ABI boundaries | Proposed |
| [0005](0005-toolchain-and-cuda-language.md) | Toolchain floor and CUDA language level | Proposed |
| [0006](0006-ci-and-supported-platforms.md) | CI lanes and supported platforms | Proposed |
| [0007](0007-project-licensing.md) | Project licensing and distribution terms | Proposed (blocked on repository owner) |
| [0020](0020-local-artifact-trust-boundary.md) | Local artifact trust boundary | Accepted |
| [0021](0021-safetensors-reader-contract.md) | Safetensors reader contract | Accepted |
| [0022](0022-llama-model-and-weight-schema.md) | Llama model and weight schema | Accepted |
| [0023](0023-model-integrity-and-fingerprint.md) | Model integrity and fingerprint | Accepted |
| [0024](0024-tokenizer-backend-and-concurrency.md) | Tokenizer backend and concurrency | Proposed (blocked) |
| [0025](0025-prompt-processing-contract.md) | Prompt processing contract | Accepted |
