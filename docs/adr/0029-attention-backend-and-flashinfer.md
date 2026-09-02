# ADR 0029: reject the recorded FlashInfer attention candidate

- Status: Rejected
- Date: 2026-09-01
- Owner: CUDA platform

## Context

M4 considered FlashInfer revision `44428003ba219c14b5473fefec8f7bfd4b72178e` for native,
ahead-of-time contiguous causal prefill/decode attention. Production acceptance requires auditing
the exact initialized recursive closure, licenses, CUDA 13/SM 89 build, native API, stream and
workspace ownership, absence of Python/JIT/runtime caches, correctness, sanitizer behavior, and
rollback.

## Decision

Reject this recorded candidate for production registration. Its gitlink is not initialized in the
qualified source checkout, so the exact source and recursive closure cannot be audited or built.
InferX does not download a substitute, carry a local patch, accept a Python/JIT path, or expose a
`flashinfer` capability. `INFERX_ENABLE_FLASHINFER=ON` fails configuration with this decision.

M4 instead owns a bounded ahead-of-time CUDA fallback for separate contiguous K/V, causal self
attention, FP32/FP16/BF16 storage with FP32 accumulation, batch 1-8, up to 64 query/KV heads, even
head dimensions 2-256, context up to 4096, prefill up to 512 total queries, and at most one decode
query per sequence. Host metadata is completely validated before KV append is launched. There is no
fallback after a launch or possible mutation.

## Alternatives

- Accepting the uninspected gitlink was rejected because a revision string alone is not dependency
  evidence.
- Fetching a newer revision opportunistically was rejected because offline reproducibility and
  review would be lost.
- Keeping Python or runtime compilation for attention was rejected by the production closure.

## Consequences

The owned fallback is a correctness/survival backend, not a long-context performance claim.
FlashInfer contributes no source, link dependency, public type, runtime cache, or capability to the
baseline. M5 must stay within the fallback envelope unless a later pin is qualified.

## Validation evidence

The repository records the exact candidate gitlink but has no initialized candidate worktree.
Configuration explicitly rejects `INFERX_ENABLE_FLASHINFER=ON`; the disabled adapter returns
`Unimplemented`. Owned attention host code passes strict syntax checking and its CUDA kernels
compile in the local diagnostic pass. CUDA 13 correctness, multi-stream, sanitizer, and benchmark
evidence remains required before supported-GPU readiness.

## Supersession

A later ADR may accept a different exact pin only after all native closure, license, offline build,
API, correctness, sanitizer, multi-stream, binary-size, and performance gates pass unchanged.
