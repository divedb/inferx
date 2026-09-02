# M4 kernel registry and warm-up

M4 separates semantic validation, deterministic capability selection, and backend preparation. No
serving request registers, tunes, inserts, or evicts a kernel.

## Kernel key

`KernelKey` schema version 1 includes the operation and phase, host/CUDA device class and SM, input/
weight/output dtypes, FP32-accumulation math mode, layout and alias IDs, logical dimensions,
attention geometry, causal flag and sequence bucket, workspace limit, actual alignment class, and
the reserved graph-compatibility value. Trailing dimensions must be zero.

It excludes pointers, allocation/stream/event/request/model IDs, filenames, clocks, opaque vendor
objects, and randomized hashes. `SerializeKernelKey` writes fixed-width little-endian fields;
`StableKernelKeyHash` hashes that same field order with FNV-1a.

## Capability and selection

A `BackendCapability` declares one backend/versioned ID and pure ranges for phase, device/SM, dtype
masks and required dtype relationships, layouts, alias modes, dimensions, attention head limits and
head-dimension multiple, alignment, deterministic class, and workspace function. Production
capabilities declare that launch does not allocate.

The registry is constructed with a fixed capacity. Registration rejects invalid or duplicate
backend/capability IDs. Freeze sorts by ascending priority, backend enum, then capability ID and
forbids later registration. Lookup is read-only and returns the first matching sorted capability,
so insertion order and pointer order cannot affect the result. A forced backend considers only that
backend and returns `Unimplemented` instead of silently falling back.

## Finite warm-up

`BuildRequiredKernelSet(LlamaOperatorSpec, OperatorEnvelope)` derives, normalizes, sorts, and
deduplicates the embedding, norm, projection, RoPE, attention, activation, residual, and logits keys
for every configured token bucket. The private model adapter converts the normalized M3 `ModelSpec`
to `LlamaOperatorSpec`; this preserves `base <- tensor <- ops` dependency direction.

`WarmupKernelSet` freezes the registry when needed, selects and prepares every required key into a
temporary fixed-capacity cache, checks prepared identity/workspace, and freezes only the complete
cache. Any miss, preparation failure, mismatch, duplicate, or capacity failure destroys the
temporary result and readiness fails. There is no partial ready state.

Vendor plans remain process-local. cuBLASLt descriptors and heuristics are created only by
`Prepare`; launch receives an already acquired aligned workspace view and an explicit stream. A
changed envelope, SM, toolkit/backend version, dtype/layout, alignment class, or workspace cap
requires preparation of a different key set.
