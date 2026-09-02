# ADR 0009: Request state machine

- Status: Proposed
- Date: 2026-08-31
- Deciding owner: repository owner (implementation proceeds provisionally)
- Seed: `docs/milestones/m1.md` sections 4, 10, 11

## Context

Request lifecycle correctness (one terminal per request, no leaked resources,
no stale-completion corruption) is the core oracle M2-M8 inherit. Ad-hoc
boolean state ("is_cancelled && !finished && ...") is exactly how those bugs
happen; the roadmap (plan section 7.2) mandates one closed transition table.

## Decision

1. **One declarative table:** `TransitionRule` array (wildcards expanded to
   concrete state/event rows) is the only transition authority; duplicate
   cells are tested against; absent cells are
   `FailedPrecondition`/`kInvalidTransition` with no side effects.
2. **Pure decision + one mutator:** `DecideTransition` (pure) + named
   resolvers for conditional destinations (reservation grants and
   completions read context counters and the event payload — the completion
   path must see its own committed outputs before commit);
   `RequestController::Prepare` materializes every fallible value (terminal
   record) and `Commit` is `noexcept`, moving prepared values only.
   The controller is the sole `RequestContext` mutator.
3. **States/events as closed vocabularies** with total-switch lowercase
   snake-case names; replay accepts only those spellings. Terminal states
   are absorbing; `Cancelling` exists so in-flight work drains before
   release.
4. **Ownership:** the registry owns contexts (hash lookup + btree arrival
   order — hash iteration is never observable); contexts own IDs/epochs
   only, never scheduler or executor state; plans/completions carry
   IDs/epochs, never pointers.
5. **Epoch/stale rules:** preemption increments the epoch (checked); a
   completion is accepted only when request/sequence/epoch/step/ticket and
   scheduled count all match and no terminal was emitted; mismatches are
   `kStaleCompletion` without side effects.
6. **Terminal atomicity:** entering a terminal state builds and marks the
   response in the same lifecycle commit; `num_scheduled_tokens` returns to
   zero on every exit path.

## Alternatives

- **Per-request boolean flags:** rejected — unbounded state-product bugs.
- **Mutating event handlers (Apply(event) edits state):** rejected — decision
  and effect must be separable for replay/invariants and noexcept commit.
- **Resolvers reading post-commit state:** impossible by construction — the
  decision precedes the commit; hence event-aware resolvers.
- **Cancel as immediate release in-flight:** rejected — releases resources
  the fake/real executor still writes; hence `Cancelling` + drain.

## Consequences

- `LifecycleCoordinator`/`ResponseSink` land with the simulator stage (M1.6)
  which fixes their concrete transaction shape; the controller/registry/FSM
  here are already the authoritative machinery.
- Adding a transition (e.g. M8 chunked prefill) means adding table rows plus
  a resolver — no handler surgery.
- The M8 chunk loop (`PrefillCompleted -> PrefillReady`) remains addable.

## Validation evidence

- `tests/unit/engine/fsm_test.cc` (label `core-unit`): happy path
  prefill→decodes→length with exact counters and terminal record;
  cancellation from every nonterminal non-in-flight state; in-flight
  cancel→drain; stale completion rejection without side effects;
  preempt/requeue epoch increment; reservation resolution by computed
  tokens; registry duplicate/order/erase-preconditions; duplicate-cell and
  name round-trip checks.

## Supersession

Superseded by a lifecycle ADR changing states, event vocabulary, or
ownership rules; replay-schema compatibility must be addressed in the same
change.
