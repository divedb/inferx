# InferX configuration (schemas v1 and v2)

The engine/simulator configuration pipeline (ADR 0011, `m1.md` section 8):
`compiled defaults < JSON file < INFERX_* environment < CLI flags`, strict
validation, immutable `EngineConfig`, byte-stable canonical JSON. The single
registry is `INFERX_CONFIG_FIELDS` in
[`include/inferx/config/parsed_config.h`](../include/inferx/config/parsed_config.h);
this table mirrors it — changes update both in one commit.

For every field `name`: environment `INFERX_<UPPERCASE_NAME>`, with dots
converted to underscores, and flag `--name`. No abbreviations or boolean-
negation aliases.

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

- Unknown JSON keys/flags, duplicate keys, wrong value types, unsupported
  nesting, oversized files, and invalid UTF-8 are errors. The sole supported
  nested object is schema-v2 `cuda` (strict parsing limits remain in
  `config/parser_limits.h`).
- An invalid value in a lower layer is an error even when a higher layer
  would replace it.
- Validation produces the immutable `EngineConfig`; `CanonicalJson()` emits
  `{"schema_version":1,...}` with lexicographic fields, decimal integers, no
  whitespace — the bytes embedded in replay headers (ADR 0012).
- M1 reserves `prompt_tokens + max_output_tokens` KV tokens at admission
  (conservative oracle; M8 owns utilization changes).

`inferx simulate validate-config` / `explain-config` (M1.6) print effective values
plus per-field provenance (`default`/`file`/`environment`/`command line`) in
stable field order.

## Schema v2 CUDA section

M1 files that do not mention a CUDA field retain their byte-identical schema-v1
canonical form. Any file, environment layer, or command-line layer that sets a
CUDA field emits schema v2 with one canonical nested `cuda` object. JSON uses
booleans for `enabled` and `enable_transfer_stream`, `null` for an automatic
device budget, and unsigned decimal integers for all other fields. Environment
and command-line overlays represent booleans as `0` or `1`; an automatic
device budget is `0` internally.

| Field | Range | Default | Rule |
|---|---|---:|---|
| `cuda.enabled` | boolean | false | True requires an `INFERX_ENABLE_CUDA` build |
| `cuda.device_id` | 0..2³²−1 | 0 | Selected ordinal must exist and qualify at context creation |
| `cuda.device_reserve_bytes` | 0..device total | 512 MiB | Excluded from allocatable budget |
| `cuda.device_budget_bytes` | null or positive bytes | null | Resolved as total minus reserve; explicit value must fit |
| `cuda.pinned_budget_bytes` | 1 MiB..4 GiB | 256 MiB | Process-wide pinned limit |
| `cuda.event_pool_slots` | 8..65,536 | 1,024 | Covers uploads and completion fences |
| `cuda.timing_event_slots` | 2..256 | 32 | Benchmark/diagnostic events only |
| `cuda.metadata_ring_slots` | 2..64 | 3 | Paired pinned/device slots |
| `cuda.metadata_slot_bytes` | 4 KiB..16 MiB | 1 MiB | Power of two |
| `cuda.staging_pool_slots` | even 2..256 | 4 | Test pipeline consumes input/output pairs |
| `cuda.staging_slot_bytes` | 4 KiB..64 MiB | 4 MiB | Power of two |
| `cuda.workspace_slots` | 1..64 | 4 | At least `plan_buffer_slots` |
| `cuda.workspace_bytes_per_slot` | 1 MiB..4 GiB | 16 MiB | Total is checked before allocation |
| `cuda.enable_transfer_stream` | boolean | true | False routes copies onto compute explicitly |

MiB/GiB use powers of 1,024; JSON stores bytes. Let
`max_test_inflight = min(metadata_ring_slots, workspace_slots,
staging_pool_slots / 2)`. Validation requires
`event_pool_slots >= metadata_ring_slots + 2 * max_test_inflight`, checks the
pinned metadata plus staging total against the pinned budget, and performs all
geometry arithmetic with overflow checks. Hardware-dependent device-total and
SM/UVA validation occurs before `CudaDeviceContext` allocation.
