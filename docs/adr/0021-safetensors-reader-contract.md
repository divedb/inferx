# ADR 0021: safetensors reader contract

- Status: Accepted
- Date: 2026-08-31
- Owner: artifacts

## Context

Safetensors has a small wire format, but accepting offsets without complete arithmetic and coverage
validation can turn a metadata parser into an out-of-bounds mapping primitive.

## Decision

InferX implements the narrow safetensors header and Hugging Face shard-index schemas directly over
the qualified simdjson adapter. It validates the little-endian header length, bounded UTF-8 JSON,
duplicate keys, exact tensor fields, dtype vocabulary, shape arithmetic, bit-packed size, ordered
contiguous payload coverage, and complete file coverage. Valid wire dtypes are represented even when
the Llama capability later rejects them.

Shard indices map every name to one safe `.safetensors` path. Each unique shard is opened once per
validation phase; every index entry and every shard tensor must match exactly, and `total_size` is
the sum of tensor payload bytes.

## Alternatives

- `safetensors-cpp` was rejected for the baseline because required shape, UTF-8-key, and offset
  validation remains caller-visible and InferX already needs strict schema parsing.
- Permissive holes, ignored tensors, and model-capability errors reported as corruption were rejected
  because they hide conversion and compatibility mistakes.

## Consequences

No `safetensors-cpp` dependency or permissive mode is used. Format corruption is `DataLoss`, resource
limits are `ResourceExhausted`, and a well-formed unsupported model dtype is `Unimplemented`. The
official Rust implementation remains the differential oracle.

## Validation evidence

`SafeTensorReaderTest.ValidatesShapeOffsetsAndMappingLifetime` and
`SafeTensorReaderTest.RejectsPayloadHoles` cover the initial valid/security cases. The official Rust
corpus and fuzz lanes remain required before closing the complete M3 gate.

## Supersession

Supersession requires a wire-compatible reader with equal or stronger differential and fuzz evidence.
