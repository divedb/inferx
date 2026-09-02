# Deterministic M1 simulator

`inferx simulate` is the CPU-only lifecycle oracle. The simulator owns immutable
configuration, a `ManualClock`, registry/controller, resource accountant,
plan pool, scheduler, fake executor, workload, response/replay buffers,
lifecycle coordinator, and invariant checker in dependency order.

No simulator path creates threads, sleeps, or reads wall time. At a timestamp
it processes:

1. shutdown, explicit cancellation, then deadlines;
2. failure injection and execution completions;
3. submissions, preemption, and requeue events;
4. at most one planning tick;
5. invariants.

Deadline expiry is inclusive (`now >= deadline`) and therefore wins over a
completion at the same instant. In-flight cancellation enters `cancelling`,
retains ownership until the matching completion drains, commits no later token,
then emits one terminal response and releases capacity.

## Commands

```text
inferx simulate validate-config --config <file>
inferx simulate explain-config --config <file> [schema overlay flags]
inferx simulate run --config <file> --workload <jsonl> --trace <jsonl>
inferx simulate replay --trace <jsonl> --output <new-jsonl>
inferx simulate check-trace --trace <jsonl>
```

Outputs are not overwritten without `--overwrite`. Exit codes are 0 success,
2 CLI/config/workload validation, 3 unexpected simulated request/executor
failure, 4 invariant/replay mismatch, and 5 output I/O failure.

The fake model is ID 0, context 32,768, token input only. Fake latency is
`base + prefill_tokens*prefill_ns + decode_sequences*decode_ns`, checked for
overflow. Tokens are a fixed function of request ID and zero-based output
position, so batch shape does not change a request's outputs.
