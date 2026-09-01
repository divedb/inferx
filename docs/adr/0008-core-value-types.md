# ADR 0008: Core value types

- Status: Proposed
- Date: 2026-08-31
- Deciding owner: repository owner (implementation proceeds provisionally)
- Seed: `docs/milestones/m1.md` sections 4, 7

## Context

Every M1+ module passes identifiers, token counts, byte sizes, and time
instants across module boundaries. The roadmap (plan section 5) forbids
unbounded/ambiguous cardinality, unchecked arithmetic, and interchangeable
identity types; M1 is where those rules become concrete types that the
scheduler, replay, and later CUDA milestones share.

## Decision

1. **Strong IDs:** one `StrongId<Tag, Rep>` template with named tag types and
   aliases per the m1.md section 7.2 table. Explicit construction, no default
   construction, no implicit integer conversion, no cross-tag conversion.
   Absence is `std::optional<Id>`. Zero is a valid value — no numeric
   sentinel; generators return `OutOfRange` before wraparound.
2. **Unit values:** `TokenCount`/`TokenOffset`/`SequenceCount`/
   `QueueCapacity` (uint32 storage) and `KvTokenCount`/`ByteCount` (uint64)
   as tag-distinct `UnitValue` types. Not arithmetic: unwrap → checked helper
   → reconstruct. `FromUint64` factories validate parsed input.
3. **Checked arithmetic:** `CheckedAdd/Sub/Mul/CeilDiv/Narrow` and
   `CheckedByteSize` in `checked_math.h` — signedness handled explicitly,
   contextual `StatusOr` failures, no wraparound reliance.
4. **Clocks:** `MonotonicTime`/`Deadline` are steady-clock nanosecond
   aliases; `Clock` interface with `SteadyClock` (production) and
   `ManualClock` (simulator/test, forward-only, overflow-rejecting). Wall
   time never enters lifecycle/replay decisions.
5. **Status:** `absl::Status`/`absl::StatusOr` remain the only error types;
   `status.h` adds the stable `ErrorReason` classification (payload
   `type.inferx.dev/error-reason`), payload helpers that reject unknown/
   malformed reasons, and the `<component>.<field>: <message>` convention.
   `status_macros.h` provides exactly `INFERX_RETURN_IF_ERROR` and
   `INFERX_ASSIGN_OR_RETURN` (single evaluation, move-only-safe,
   pre-declared lvalue assignment only).

## Alternatives

- **Raw integer IDs/counts (typedef int):** rejected — interchangeable by
  construction; every scheduler/replay bug class the roadmap warns about.
- **Default-constructible IDs with an invalid sentinel:** rejected — zero is
  valid for ranks/devices/token IDs; a sentinel overloads a valid value.
- **Arithmetic unit types (operator+ on TokenCount):** rejected — hides
  overflow and mixed-unit bugs at the type system's weakest point.
- **A second InferX error type/enum:** rejected (ADR 0002); the reason
  classification accompanies statuses instead of replacing codes.
- **Wall-clock timestamps in decisions:** rejected — nondeterministic across
  machines; replay equality (m1.md section 15) would be undefined.

## Consequences

- `inferx::base` publicly depends on Abseil (Status/hash/stringify), so the
  installed package requires `find_dependency(absl CONFIG)`; the M0 consumer
  fixture strategy (pinned Abseil in the prefix) becomes the default
  install path.
- Compile errors, not runtime checks, catch unit/ID mix-ups.
- Adding an ID or unit requires a named tag plus boundary tests, not a raw
  alias.

## Validation evidence

- `tests/unit/base/m1_{id,token,checked_math,clock,status,status_macros}_test.cc`
  (label `m1-unit`): construction boundaries, zero/max round trips, generator
  overflow, every signed/unsigned branch, payload validation, macro
  single-evaluation and move-only behavior.
- Header self-containment probes cover all new headers; consumer test links
  the exported `inferx::base` against the prefix Abseil.

## Supersession

Superseded only by a new ADR changing representation, sentinel policy, or the
status convention; replay-schema compatibility (ErrorReason numbering) must be
addressed in the same change.
