# InferX configuration (schema v1)

The engine/simulator configuration pipeline (ADR 0011, `m1.md` section 8):
`compiled defaults < JSON file < INFERX_* environment < CLI flags`, strict
validation, immutable `EngineConfig`, byte-stable canonical JSON. The single
registry is `INFERX_CONFIG_FIELDS` in
[`include/inferx/config/parsed_config.h`](../include/inferx/config/parsed_config.h);
this table mirrors it — changes update both in one commit.

For every field `name`: environment `INFERX_<UPPERCASE_NAME>`, flag `--name`.
No abbreviations, no boolean-negation aliases.

## Schema fields

| Field | Range | Default | Rule |
|---|---|---:|---|
| `max_active_sequences` | 1..1,000,000 | 256 | Must be ≤ `max_queued_requests` |
| `max_model_tokens` | 2..2³¹−1 | 4,352 | Must not exceed model capabilities |
| `max_output_tokens` | 1..2³¹−1 | 256 | Per-request cap |
| `max_prompt_tokens` | 1..2³¹−1 | 4,096 | Must be ≤ `max_scheduled_tokens_per_step` (M1; chunking is M8) |
| `max_queued_requests` | 1..1,000,000 | 4,096 | Bounds all live admitted requests |
| `max_scheduled_tokens_per_step` | 1..2³¹−1 | 4,096 | Prefill + decode token budget |
| `max_sequences_per_step` | 1..16,384 | 32 | Must be ≤ min(`max_active_sequences`, 16,384) |
| `max_simulation_events` | 1..100,000,000 | 10,000,000 | Deadlock/runaway bound |
| `plan_buffer_slots` | 1..64 | 1 | In-flight fake steps |
| `response_channel_capacity` | ≥1 | 4,096 | Reference channel config |
| `simulated_kv_token_capacity` | ≥1 (64-bit) | 1,114,112 | = 256 × 4,352 full-reservation reference |
| `submission_channel_capacity` | ≥1 | 4,096 | Reference channel config |
| `fake_base_latency_ns` | ≥1 | 1,000 | Fake executor cost; latency products are overflow-checked against token/sequence budgets |
| `fake_prefill_latency_per_token_ns` | ≥0 | 100 | Checked multiply by tokens |
| `fake_decode_latency_per_sequence_ns` | ≥0 | 100 | Checked multiply by sequences |

## Semantics summary

- Unknown JSON keys/flags, duplicate keys, non-integer values, nesting,
  oversized files, invalid UTF-8: errors (strict parsing; m1.md section 8.4
  lists the byte/depth/member limits in `config/parser_limits.h`).
- An invalid value in a lower layer is an error even when a higher layer
  would replace it.
- Validation produces the immutable `EngineConfig`; `CanonicalJson()` emits
  `{"schema_version":1,...}` with lexicographic fields, decimal integers, no
  whitespace — the bytes embedded in replay headers (ADR 0012).
- M1 reserves `prompt_tokens + max_output_tokens` KV tokens at admission
  (conservative oracle; M8 owns utilization changes).

`inferx-sim validate-config` / `explain-config` (M1.6) print effective values
plus per-field provenance (`default`/`file`/`environment`/`command line`) in
stable field order.
