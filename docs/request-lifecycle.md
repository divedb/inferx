# Request lifecycle (M1)

The closed transition table (ADR 0009; `m1.md` section 11). The executable
authority is `TransitionRules()` in
`src/engine/request_state_machine.cc`; this table explains the conditional
guards, while the generated inventory at the end of this file comes directly
from `inferx inspect --fsm-schema` and is checked by CTest.

States: `received tokenizing queued reserving prefill_ready prefilling
decode_ready decoding preempted cancelling finishing finished cancelled
failed` — `finished`/`cancelled`/`failed` are terminal and absorbing.

| Current | Event | Guard | Next |
|---|---|---|---|
| received | start_tokenization | text input/test processor | tokenizing |
| received | input_ready | validated token input | queued |
| tokenizing | tokenization_succeeded | nonempty valid tokens | queued |
| tokenizing | tokenization_failed | terminal status supplied | **failed** |
| queued | begin_reservation | scheduler selected request | reserving |
| reserving | reservation_granted | reservation exists | prefill_ready / decode_ready (by computed tokens) |
| reserving | reservation_deferred | no resource change | queued |
| prefill_ready | submit_prefill | no in-flight work; matching plan item | prefilling |
| prefilling | prefill_completed | matching step/ticket/epoch; success | decode_ready / finishing (no output remains) |
| decode_ready | submit_decode | output remains; matching plan item | decoding |
| decoding | decode_completed | matching step/ticket/epoch; success | decode_ready / finishing |
| prefill_ready, decode_ready | preempt | no in-flight work | preempted (reservation released) |
| preempted | requeue | epoch increment succeeds | queued (epoch incremented) |
| any nonterminal non-in-flight except finishing | cancel_requested, deadline_expired | none | **cancelled** |
| prefilling, decoding | cancel_requested, deadline_expired | matching work remains owned | cancelling |
| cancelling | in_flight_drained | matching completion; no commit | **cancelled** |
| prefilling, decoding | execution_failed | matching failure completion | **failed** |
| any nonterminal non-in-flight | fatal_error | terminal status supplied | **failed** |
| finishing | terminal_emitted | exactly one terminal response built | **finished** |

Rules fixed by the controller (`RequestController::Prepare`/`Commit`):

- terminal states are absorbing; any event there is `FailedPrecondition`;
- completions match request, sequence, epoch, step, ticket, and scheduled
  token count, or are rejected as `kStaleCompletion` with no side effects;
- entering a terminal state builds and marks exactly one terminal response
  in the same commit and zeroes outstanding scheduled tokens;
- prefill marks all prompt tokens computed and commits the first synthetic
  output; each decode marks one token computed and commits one output
  (counter semantics in `m1.md` section 10.1).

<!-- BEGIN GENERATED FSM -->

## Generated transition inventory

Generated from the compiled declarative table; do not edit this section manually.

| From | Event | To | Effects | Terminal |
|---|---|---|---|---|
| `received` | `start_tokenization` | `tokenizing` | none | no |
| `received` | `input_ready` | `queued` | none | no |
| `tokenizing` | `tokenization_succeeded` | `queued` | none | no |
| `tokenizing` | `tokenization_failed` | `failed` | none | yes |
| `queued` | `begin_reservation` | `reserving` | none | no |
| `reserving` | `reservation_granted` | `conditional` | none | no |
| `reserving` | `reservation_deferred` | `queued` | none | no |
| `prefill_ready` | `submit_prefill` | `prefilling` | none | no |
| `prefilling` | `prefill_completed` | `conditional` | clear-in-flight | no |
| `decode_ready` | `submit_decode` | `decoding` | none | no |
| `prefill_ready` | `stop_matched` | `finishing` | none | no |
| `decode_ready` | `stop_matched` | `finishing` | none | no |
| `decoding` | `decode_completed` | `conditional` | clear-in-flight | no |
| `prefill_ready` | `preempt` | `preempted` | release-reservation | no |
| `decode_ready` | `preempt` | `preempted` | release-reservation | no |
| `preempted` | `requeue` | `queued` | increment-epoch | no |
| `received` | `cancel_requested` | `cancelled` | none | yes |
| `received` | `deadline_expired` | `cancelled` | none | yes |
| `tokenizing` | `cancel_requested` | `cancelled` | none | yes |
| `tokenizing` | `deadline_expired` | `cancelled` | none | yes |
| `queued` | `cancel_requested` | `cancelled` | none | yes |
| `queued` | `deadline_expired` | `cancelled` | none | yes |
| `reserving` | `cancel_requested` | `cancelled` | none | yes |
| `reserving` | `deadline_expired` | `cancelled` | none | yes |
| `prefill_ready` | `cancel_requested` | `cancelled` | release-reservation | yes |
| `prefill_ready` | `deadline_expired` | `cancelled` | release-reservation | yes |
| `decode_ready` | `cancel_requested` | `cancelled` | release-reservation | yes |
| `decode_ready` | `deadline_expired` | `cancelled` | release-reservation | yes |
| `preempted` | `cancel_requested` | `cancelled` | none | yes |
| `preempted` | `deadline_expired` | `cancelled` | none | yes |
| `prefilling` | `cancel_requested` | `cancelling` | none | no |
| `prefilling` | `deadline_expired` | `cancelling` | none | no |
| `decoding` | `cancel_requested` | `cancelling` | none | no |
| `decoding` | `deadline_expired` | `cancelling` | none | no |
| `cancelling` | `in_flight_drained` | `cancelled` | release-reservation, clear-in-flight | yes |
| `prefilling` | `execution_failed` | `failed` | release-reservation, clear-in-flight | yes |
| `decoding` | `execution_failed` | `failed` | release-reservation, clear-in-flight | yes |
| `received` | `fatal_error` | `failed` | none | yes |
| `tokenizing` | `fatal_error` | `failed` | none | yes |
| `queued` | `fatal_error` | `failed` | none | yes |
| `reserving` | `fatal_error` | `failed` | none | yes |
| `prefill_ready` | `fatal_error` | `failed` | release-reservation | yes |
| `decode_ready` | `fatal_error` | `failed` | release-reservation | yes |
| `preempted` | `fatal_error` | `failed` | none | yes |
| `finishing` | `terminal_emitted` | `finished` | release-reservation | yes |

<!-- END GENERATED FSM -->
