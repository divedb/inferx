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
| [0005](0005-toolchain-and-cuda-language.md) | Toolchain floor and CUDA language level | Superseded by [ADR 0020](0020-cuda-13-toolchain-and-m2-qualification.md) |
| [0006](0006-ci-and-supported-platforms.md) | CI lanes and supported platforms | Proposed |
| [0007](0007-project-licensing.md) | Project licensing and distribution terms | Proposed — **blocked on repository owner decision**; no coding agent may |
| [0008](0008-core-value-types.md) | Core value types | Proposed |
| [0009](0009-request-state-machine.md) | Request state machine | Proposed |
| [0010](0010-scheduler-resource-transactions.md) | Deterministic scheduler and resource transactions | Proposed |
| [0011](0011-configuration-schema.md) | Configuration schema and pipeline | Proposed |
| [0012](0012-deterministic-simulator-replay.md) | Deterministic simulator and canonical replay | Proposed |
| [0013](0013-bounded-channel.md) | Bounded channel semantics | Proposed |
| [0014](0014-tensor-buffer-contract.md) | Tensor and buffer contract | Accepted |
| [0015](0015-memory-accounting-and-pools.md) | Memory accounting and fixed pools | Accepted |
| [0016](0016-cuda-context-and-error-scope.md) | CUDA context ownership and error scope | Accepted |
| [0017](0017-stream-event-fence-contract.md) | Stream, event, and fence contract | Accepted |
| [0018](0018-metadata-and-workspace-lifetime.md) | Metadata and workspace lifetime | Accepted |
| [0019](0019-m2-gpu-qualification.md) | M2 GPU qualification | Superseded by [ADR 0020](0020-cuda-13-toolchain-and-m2-qualification.md) |
| [0020](0020-cuda-13-toolchain-and-m2-qualification.md) | CUDA 13.0 toolchain and M2 GPU qualification | Accepted |
| [0021](0021-safetensors-reader-contract.md) | Safetensors reader contract | Accepted |
| [0022](0022-llama-model-and-weight-schema.md) | Llama model and weight schema | Accepted |
| [0023](0023-model-integrity-and-fingerprint.md) | Model integrity and fingerprint | Accepted |
| [0024](0024-tokenizer-backend-and-concurrency.md) | Tokenizer backend and concurrency | Proposed (blocked on an approved replacement pin) |
| [0025](0025-prompt-processing-contract.md) | Prompt processing contract | Accepted |
| [0026](0026-operator-contracts-and-numeric-policy.md) | Operator contracts and numeric policy | Accepted |
| [0027](0027-kernel-registry-and-warmup.md) | Kernel registry and warm-up | Accepted |
| [0028](0028-cublaslt-algorithm-and-workspace.md) | CuBLASLt algorithm and workspace policy | Accepted |
| [0029](0029-attention-backend-and-flashinfer.md) | Reject the recorded FlashInfer attention candidate | Rejected |
| [0030](0030-cuda-glue-kernel-policy.md) | Owned CUDA glue kernel policy | Accepted |
| [0031](0031-kernels-backend-architecture.md) | Kernels backend architecture and provider chain | Accepted |
| [0032](0032-external-kernel-providers.md) | External kernel provider qualification at the pinned revisions | Accepted (provisional; evidence recorded below) |
| [0033](0033-hugging-face-model-resolution.md) | Hugging Face model resolution and cache handoff | Accepted |
| [0034](0034-local-artifact-trust-boundary.md) | Local artifact trust boundary | Accepted |
