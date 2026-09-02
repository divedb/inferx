# Qualification report: nlohmann/json

- Manifest entry: `nlohmann-json` — feature `tokenization` (M3), owner `tokenization`
- Pin: `734fd305a19995673c5b83b4a30c06a96ff5c9b0` (submodule `third_party/nlohmann-json`)
- License: MIT (`LICENSE.MIT` at the pin)
- Decision: **Approved (ADR 0024, 2026-09-02)**

## Use and boundary

Single-header JSON DOM used **privately inside the tokenizer vendor package**
(`third_party/tokenizer`): the adapted divedb config parsing and minja's API
both require it. Consumed via `SYSTEM` include directories scoped to the
vendor target only; never appears in an InferX public header, and InferX's
own artifact/config parsing continues to use simdjson exclusively.

## Why a second JSON library is acceptable

ADR 0024's qualification criteria explicitly anticipate "minja/nlohmann if
chat rendering retains them": reusing the adapted tokenizer code as-is is
the point of the vendor package, and rewiring it onto simdjson would be a
rewrite of qualified code. The dependency is confined to one non-installed
target and recorded here.

## Security

Header-only; parses only checkpoint-controlled tokenizer/config/chat bytes
already inside the rooted artifact session. Malformed input flows through
the package's non-throwing parse wrapper (`allow_exceptions=false`) and the
engine's own error-returning construction; the subprocess fuzz corpus covers
this path.
