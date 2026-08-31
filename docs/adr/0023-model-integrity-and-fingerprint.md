# ADR 0023: model integrity and fingerprint

- Status: Accepted
- Date: 2026-08-31
- Owner: artifacts/model identity

## Context

Paths, mtimes, and process hash functions are not durable model identity. A package identity must
cover bytes and normalized semantics while remaining stable when the same immutable package moves.

## Decision

InferX uses official BLAKE3 C 1.8.7 at commit
`f3149ec5bb5449af877ba20377a11008ff499fa2`. `Digest256` is a fixed 32-byte value with lowercase hex
and constant-time equality. Every consumed artifact is streamed from its opened fd and checked for
mutation.

An optional manifest v1 declares the selected weight entry, opaque revision, exact size, and digest
of every consumed file. It neither lists nor hashes itself. A manifest mismatch is fatal `DataLoss`.

The base model fingerprint is a BLAKE3-256, domain-separated, length-delimited binary record stream
with schema version 1. It includes normalized model semantics, plan schema, qualified tokenizer
capability, sorted artifact content digests, manifest revision, source layout, quantization, adapter,
and RoPE policy. Paths outside the root, inode data, timestamps, and logging flags are excluded.

## Alternatives

- SHA variants were not selected because BLAKE3 is already the roadmap choice for M6/M11 and has a
  qualified streaming C implementation.
- Hashing headers, paths, or canonical JSON alone was rejected because it omits weight bytes or inert
  artifact changes.
- Publishing a partial identity without tokenizer semantics was rejected as namespace ambiguity.

## Consequences

The implemented fingerprint primitive cannot publish a `ModelFingerprint` for an inspected package
until a qualified tokenizer provides canonical metadata. The CLI says `unavailable` instead of
minting a misleading partial identity.

## Validation evidence

`DigestTest.MatchesOfficialBlake3Vectors`, manifest/gitlink validation, file identity checks, and
deterministic record-order unit coverage are the current evidence. Full fingerprint mutation tests
activate with the tokenizer backend.

## Supersession

Any preimage change increments the fingerprint schema; downstream replica/cache namespaces derive
from rather than overwrite the base fingerprint.
