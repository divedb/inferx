# ADR 0042: Retire the M1 simulator and `simulate` CLI

- Status: Accepted
- Date: 2026-09-03
- Owner: repository owner
- Supersedes: [ADR 0012](0012-deterministic-simulator-replay.md) (rejected)

## Context

ADR 0012 provisioned a CPU-only deterministic simulator (`simulator/`,
`inferx simulate`) as the executable oracle for lifecycle, cancellation,
resource ownership, and replay determinism before any real execution existed.
That role is over: M5 delivered a real CPU execution path
(`SingleRequestRunner`, `CpuExecutionBackend`) with its own contracts,
configuration (`execution.*`, schema v3), and tests, and the roadmap's GPU
backends validate against real execution, not fake latency. The simulator had
exactly one production entry point (the `simulate` dispatcher path), depended
one-way on core libraries, and consumed the only simulator-specific
configuration fields — while future milestones (M8 simulation tooling, M11/M12
CPU policy simulation) kept accruing obligations against it. Maintaining a
second parallel execution model costs build time, CI lanes, config surface,
and spec complexity for a gate that no longer guards anything the M5+ test
suite does not.

## Decision

1. The `simulator/` library, the `inferx simulate` command tree
   (`validate-config`, `explain-config`, `run`, `replay`, `check-trace`), the
   simulator/replay/stress tests and benchmark, and the nightly
   `inferx_simulator_stress` CI job are removed. ADR 0012 is rejected.
2. The simulator-only configuration fields (`max_simulation_events`,
   `simulated_kv_token_capacity`, `fake_base_latency_ns`,
   `fake_prefill_latency_per_token_ns`,
   `fake_decode_latency_per_sequence_ns`), `BuildCapabilities::simulator`,
   and `FinishReason::kSimulatedEos` are removed from the code and schema.
   Config files that set them now fail with `unknown field`, per ADR 0011's
   strict boundary. The canonical form of schemas v1–v3 changes accordingly;
   no reader of the old bytes remains.
3. The configuration pipeline itself (ADR 0011) stays: it is the designated
   home of the real engine's `execution.*`/`cuda.*` sections, currently
   exercised by config unit tests until the engine consumes them directly.
4. `ManualClock` stays in `inferx::base` as a generic test double.
5. Future milestones must not assume a CPU scheduler simulator, replay
   oracle, or simulation tooling; any such need must be re-proposed on real
   execution evidence.

## Alternatives

- Keep the simulator as an offline policy-lab for M8/M11 scheduling work:
  rejected — the fake-execution model cannot rank policies for real
  throughput, and the maintenance cost of a second execution model outweighs
  exploratory value; policy comparison can be rebuilt against real execution
  when M8 needs it.
- Keep the binary but freeze it (no CI, no spec obligations): rejected — a
  growing core makes frozen simulators rot silently while still shipping
  config surface and CLI documentation.
- Remove only the CLI command, keep the library: rejected — the library had
  no other consumer; the dispatcher was the single core→simulator edge.

## Consequences

- `docs/milestones/m1.md` stands as a historical record; its
  simulator/replay halves are void. Milestones m7, m8, m10, m11, and m12 and
  `docs/plan.md` lose their simulator-based prerequisites, gates, and tooling
  plans and must be re-planned on real-execution evidence.
- The core CTest categories `core-integration`, `core-failure`, and
  `core-stress` become empty and leave the required-label CI check;
  `core-correctness` is carried by `fsm_docs_current`.
- Replay traces and simulator config files from earlier builds are obsolete;
  `simulated_eos` values in stored events are no longer parseable.
- Configuration provenance tooling (`explain-config`-style) may return later
  as part of a real `inferx` diagnostics command.

## Validation evidence

- `cmake --preset dev-gcc && cmake --build --preset dev-gcc` and
  `ctest --preset dev-gcc` pass (209/209) with the subsystem removed.
- `tools/ci/check_test_labels.py` passes with the trimmed required set.
- `grep -rn "simulat" src/ include/ tests/ benchmarks/ tools/` finds no
  simulator references beyond unrelated labels.

## Supersession

Superseded by a future ADR if scheduling-policy development again requires an
offline deterministic execution model.
