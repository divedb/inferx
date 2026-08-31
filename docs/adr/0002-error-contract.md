# ADR 0002: Cross-module error contract

- Status: Proposed
- Date: 2026-08-30
- Deciding owner: repository owner (implementation proceeds provisionally)
- Seed: `docs/milestones/m0.md` section 4; `docs/plan.md` section 5.2

## Context

Every milestone needs to return failures across module boundaries: artifact validation,
tokenization, scheduling, CUDA launches, collectives, and HTTP translation. A parallel
error taxonomy (custom result enums, exceptions, `std::optional` + side-channel logs)
would let each subsystem drift, force lossy conversions at boundaries, and make
per-request failure semantics (plan section 5.2) impossible to enforce uniformly.
Abseil is already the pinned utility baseline candidate for M0.

## Decision

1. `absl::Status` and `absl::StatusOr<T>` are the single cross-module error
   representation inside InferX.
2. No exception crosses any InferX-owned boundary (public API, engine, plugin/backend
   seam). Adapters catch dependency exceptions at the seam and convert (ADR 0004).
3. Failure classes from plan section 5.2 (invalid input, cancellation, capacity,
   corrupt artifact, backend error, device loss, rank divergence, KV transfer,
   telemetry) map to canonical `absl::StatusCode` values plus structured payloads
   (component, device, rank, request ID, retryability). Canonical mapping table lives
   with the M1 `base` status work; M0 only fixes the type choice.
4. User-visible messages are sanitized; detailed causes go to logs/traces, not API
   responses.
5. In M0, Abseil stays **private** to InferX build internals (qualification test only).
   M1 exposes `Status` through `inferx::base` and then the installed-consumer rules of
   `docs/milestones/m0.md` section 6.7 (pinned Abseil package in the prefix,
   `find_dependency(absl CONFIG)`) become mandatory.

## Alternatives

- **`std::expected<T, Error>` everywhere:** rejected as primary contract; no canonical
   code taxonomy, no payload/message ecosystem, and C++23 monadic ergonomics do not
   cover the logging/telemetry integration. `std::expected` may still appear for
   tight-scope local results.
- **Exceptions:** rejected; crossing thread pools, GPU completion paths, and plugin
   seams with exceptions contradicts plan section 5.1 and ADR 0004.
- **Custom Status clone:** rejected; reinvents `absl::Status` with less tooling.

## Consequences

- Abseil becomes a required dependency of `inferx::base` from M1; the M0
  installed-dependency consumer fixture (`tests/consumer_absl`) proves the packaging
  path early.
- All new public functions return `Status`/`StatusOr` or are declared `noexcept` with
  documented failure behavior; clang-tidy reviews enforce the convention as it lands.
- Adapters for CUDA/tokenizer/HTTP must translate error codes; translation tables are
  owned by the adapter module, not callers.

## Validation evidence

- `inferx_absl_qualification_test` (CTest label `dependency`) constructs, inspects,
   and propagates `absl::Status`/`StatusOr` against the pinned Abseil revision.
- `tests/consumer_absl` installs the pinned Abseil into a temporary prefix and
   consumes it via `find_package(absl CONFIG)` without the source checkout on any
   include path (expected-failure fixture covers the missing-package path).

## Supersession

Superseded by a new ADR only with evidence that Abseil fails a qualification gate
(license, toolchain, size) or the error taxonomy proves insufficient; the successor
must define the migration for every existing signature.
