# Request lifecycle (M1)

The closed transition table (ADR 0009; `m1.md` section 11). The executable
authority is `TransitionRules()` in
`src/engine/request_state_machine.cc`; this table mirrors it — changes
update both in one commit. (The `inferx_fsm_schema` generator that renders
this file from the compiled array lands with M1.7, per the milestone's
tooling stage; until then the duplicate-cell and coverage tests guard drift.)

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
