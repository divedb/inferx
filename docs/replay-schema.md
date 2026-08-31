# Replay schema v1

Replay is canonical UTF-8 JSON Lines. Every record begins with
`schema_version`, `record_type`, and contiguous `ordinal` in that order. The
closed record sequence is `header`, zero or more `workload_event` records,
derived lifecycle records, and one final `footer`.

| Record | Ordered payload fields |
|---|---|
| `header` | `inferx_version`, `config`, `fake_model`, `workload_event_count` |
| `workload_event` | `at_ns`, `event_type`, canonical event-specific fields |
| `request_transition` | `at_ns`, `request_id`, `sequence_id`, `epoch`, `from`, `event`, `to`, `status_code`, `error_reason` |
| `resource_change` | `at_ns`, `action`, `reservation_id`, `request_id`, deltas, used totals |
| `step_plan` | timestamp/step/model/slot/generation, aggregate counts, ordered `sequences` |
| `execution_completion` | timestamp/ticket/step/request/sequence/epoch/kind/range/status/reason/token |
| `response_event` | timestamp/request/kind/position/token/finish/status/reason/usage |
| `idle` | `at_ns`, `idle_reason`, `next_event_ns` |
| `footer` | final status/reason, counters, final ownership, `invariants_ok` |

Variant fields that do not apply are JSON `null`. IDs, counts, timestamps, and
durations are decimal integers. Status messages are excluded; only the stable
numeric Abseil code and closed `ErrorReason` name are serialized. Arrays retain
scheduler order. Pointer values, build paths, wall time, locales, environment
dumps, and unordered-container iteration cannot enter the trace.

The checked example is
[`tests/correctness/replay/data/basic_trace.jsonl`](../tests/correctness/replay/data/basic_trace.jsonl).
`check-trace` rejects invalid JSON, unknown versions/types/fields, duplicate or
missing fields, non-contiguous ordinals, truncation, records after the footer,
and configured byte limits. `replay` reconstructs canonical config/workload,
reruns, writes a separate file, and compares exact bytes.
