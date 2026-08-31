# ADR 0024: tokenizer backend and concurrency

- Status: Proposed (blocked on an approved replacement pin)
- Date: 2026-08-31
- Owner: tokenization

## Context and audit result

The pinned `divedb/tokenizer` revision is
`f109b7aef148dd4866a3dae7a8e5a6d221f95c75`. Its logical ownership model is suitable—one
thread-affine handle per worker—but its build unconditionally adds duplicate dependencies and a
network/OpenSSL closure. Its Rust shim has `unwrap()` paths reachable from malformed bytes and it
does not expose upstream streaming decode state.

## Decision

Reject that revision unchanged. `INFERX_ENABLE_TOKENIZATION` defaults off and fails configuration
when enabled, naming the missing qualifications. InferX will not implement BPE, Unigram,
SentencePiece, Unicode normalization, byte fallback, or incremental suffix heuristics itself.

The replacement revision must provide parent dependencies, local-only mode, error-returning FFI,
owned output bytes, externalized upstream `step_decode_stream` state, disabled internal
oversubscription, and a complete recursive license/SBOM record. It is approved only after exact
10,000-case Hugging Face differential, malformed-input subprocess, and TSan pool tests.

## Alternatives

- Embedding the current CMake project unchanged was rejected for duplicate targets, network closure,
  filesystem mutation, and abort behavior.
- Catching C++ exceptions was rejected because Rust abort/panic paths do not become recoverable.
- A home-grown tokenizer or repeated-prefix incremental decode was rejected for fidelity and
  already-emitted-byte correctness.

## Consequences

M3 is not complete and no `ValidatedModelPackage`, serving tokenizer pool, prompt processor, or base
fingerprint is published yet. Artifact/model inspection remains independently useful and honest.

## Validation evidence

The source audit is recorded in `docs/dependencies/tokenizer.md`; configure-time failure
`failure_tokenizer_unqualified` proves the rejected candidate cannot be enabled accidentally. This
ADR cannot become Accepted until the named differential, subprocess, and TSan evidence exists.

## Supersession

Accept this ADR by naming the exact replacement revision and closure, or supersede it with another
backend that satisfies the same public contract and corpus.
