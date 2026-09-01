# ADR 0027: kernel registry and warm-up

- Status: Accepted
- Date: 2026-09-01
- Owner: operator runtime

## Context

Backend choice must not depend on registration order, pointer values, request timing, or an
unbounded online tuning cache. A model must fail readiness before serving if its finite operation
envelope cannot be prepared.

## Decision

`KernelKey` schema version 1 is a fixed-width durable value with closed enums, normalized trailing
dimensions, no pointers or request identity, little-endian serialization, and stable FNV-1a hashing.
`BackendCapability` is an immutable predicate over device/SM, phase, dtype relationships, layouts,
alias mode, dimensions, attention geometry, alignment, and workspace.

Registration has a fixed capacity. `Freeze()` validates and sorts by numeric priority, backend, and
capability ID, then forbids mutation. Selection chooses the first matching sorted capability. Forced
selection inspects only the requested backend and never falls back. Prepared plans have a separate
fixed-capacity cache; warm-up builds a temporary complete cache, freezes it only after every key is
prepared, and discards it on any failure. Live lookup performs no insertion, eviction, heuristic
search, or allocation.

The public `inferx::ops` warm-up input is `LlamaOperatorSpec`, keeping ops independent of model and
artifact code. The private model adapter translates M3 `ModelSpec` into that value before deriving
the finite set.

## Alternatives

- An unordered registry was rejected because iteration and hash seeding are not durable policy.
- Lazy first-request preparation and LRU eviction were rejected because they add latency,
  allocation, and failure after readiness.
- Persisting opaque vendor algorithms was rejected because their validity is toolkit/driver/SM
  scoped.

## Consequences

Every admitted shape consumes a bounded prepared entry and must be present before readiness. A
changed envelope or dependency/runtime version requires a new warm-up. Later offline tuning may
change committed priorities, but live traffic cannot race to choose a winner.

## Validation evidence

`M4RegistryTest.SelectionIsIndependentOfRegistrationOrder` proves the stable tie-break.
`M4RegistryTest.RequiredReferenceSetWarmsTransactionally` derives all keys for two token buckets,
forces the reference backend, freezes the registry/cache, and verifies every lookup. The test passed
under the GCC C++23 M4 unit lane on 2026-09-01.

## Supersession

Cache persistence or dynamic shape admission requires a new ADR defining versioning, bounds,
readiness semantics, and deterministic eviction or proving that no eviction exists.
