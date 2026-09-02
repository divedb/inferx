# ADR 0011: Configuration schema and pipeline

- Status: Proposed
- Date: 2026-08-31
- Deciding owner: repository owner (implementation proceeds provisionally)
- Seed: `docs/milestones/m1.md` sections 4, 8

## Context

The engine and simulator need one validated, immutable configuration with
clear provenance: where each value came from, whether unknown fields are
trusted, and a byte-stable canonical form that replay can embed. Mutable
global config, silent unknown-field tolerance, and floating-point durations
would break determinism and debuggability.

## Decision

1. **Layers:** compiled defaults < JSON file < documented `INFERX_*`
   environment < CLI flags. Later layers override only fields they contain.
   An invalid value in any layer is an error even if a later layer would
   replace it (broken deployment inputs surface instead of being masked).
2. **Strictness:** unknown JSON keys/flags, duplicate JSON keys, non-integer
   values, nesting, oversized files, and invalid UTF-8 are errors. Schema-v1
   safety constants live in `config/parser_limits.h`.
3. **Representations:** `ParsedConfig` (value + `ConfigSource` provenance per
   field) → `ValidateConfig(parsed, BuildCapabilities, ModelCapabilities)` →
   immutable `EngineConfig` (private construction, read-only accessors) plus
   `CanonicalJson()`.
4. **Canonical form:** UTF-8 JSON, schema version first, fixed lexicographic
   field order, decimal integers only, no insignificant whitespace. The
   replay header embeds and byte-compares these exact bytes; no persistent
   short hash is invented (a BLAKE3 display fingerprint may come later
   without changing canonical identity).
5. **Parsing:** simdjson (pinned v4.6.5, qualified per
   `docs/dependencies/simdjson.md`) behind a private adapter in
   `src/config/internal/`; dependency exceptions, if any, are caught at the
   adapter and translated with field context. No simdjson type appears in a
   public header.
6. **M1 conservative KV policy:** `prompt_tokens + max_output_tokens` tokens
   are reserved at admission; requests that can never fit are rejected before
   queueing. Utilization changes belong to M8, behind their own ADR.

## Alternatives

- **Mutable global/singleton config:** rejected — untracked mutation, no
  provenance, hostile to the single-writer coordinator model.
- **Unknown-field tolerance (forward compat):** rejected — typos become
  silent misconfiguration; versioned schemas handle evolution instead.
- **Floating-point durations/counts:** rejected — canonical byte equality
  across compilers/libm is unattainable; nanosecond integers are exact.
- **`std::hash`/`absl::Hash` as persistent config identity:** rejected —
  process-unstable (m1.md section 2.2); canonical bytes are the identity.
- **Relative/monotonic config time:** N/A (config carries no wall time).

## Consequences

- Adding a field touches one registry (`INFERX_CONFIG_FIELDS`), the range
  table in validation, and the documented table in `docs/configuration.md`;
  the canonical writer and env/CLI spellings derive automatically.
- Environment spelling is mechanical (`INFERX_<UPPERCASE_NAME>`); no
  abbreviations or boolean negation aliases.
- `explain-config` (M1.6 CLI) prints effective values + sources in stable
  field order, straight from `ParsedConfig` provenance.

## Validation evidence

- `tests/unit/config/config_test.cc` (label `core-unit`): defaults,
  precedence, provenance, unknown/duplicate fields, integer strictness,
  parser limits, every range and cross-field rule, latency overflow,
  capabilities mismatch, and canonical byte-equality across reordered input
  keys and equivalent overlay combinations.

## Supersession

Superseded by a schema-version ADR (field additions beyond v1 semantics,
floating point, or layer changes); replay compatibility must be addressed in
the same change.
