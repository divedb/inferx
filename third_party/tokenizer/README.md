# InferX tokenizer vendor package (ADR 0024)

An **owned, local-only** tokenizer backend for InferX. Not a submodule and
not an unmodified upstream tree: this directory is a qualified adaptation
with recorded provenance, and it is the only tokenizer implementation in
InferX.

## Provenance and license

- `LICENSE.divedb-tokenizer` — the MIT license of the adapted upstream.
- The C++ layer under `include/tokenizer/` and `src/` is adapted from
  [divedb/tokenizer](https://github.com/divedb/tokenizer) at revision
  `f109b7aef148dd4866a3dae7a8e5a6d221f95c75` (audited 2026-08-31; see
  `docs/dependencies/tokenizer.md`). Every adapted file carries a header
  note where it diverges from upstream.
- `rust/` is **new code owned by InferX** (`inferx-tokenizers-c`): an
  error-returning C ABI over the official Hugging Face Rust `tokenizers`
  engine. It replaces the upstream tokenizers-cpp Rust shim, which aborted
  the process through `unwrap()` on malformed input, aliased decode results
  to handle-owned scratch, and exposed no streaming decode state.

## Composition rules

- Parent-provided Abseil only; nothing is added, force-set, or symlinked.
- No Hub/cache/curl/OpenSSL/SentencePiece/batch/runtime-token-mutation code.
- The loading path consumes checkpoint **bytes** (`LocalArtifacts`); this
  package performs no filesystem or network access anywhere.
- The Rust crate builds **offline** from the vendored crates.io closure
  (`rust/vendor/`, pinned by `rust/Cargo.lock`; `rust/.cargo/config.toml`
  sets `[net] offline = true`).
- nlohmann/json and minja are repository submodules
  (`third_party/nlohmann-json`, `third_party/minja`) and stay private to
  this target's sources.

## Layout

```
CMakeLists.txt          cargo staticlib + C++ wrapper library
include/tokenizer/      adapted public headers (namespace tokenizer)
src/                    adapted + new implementations
src/backend/inferx_abi.h  the owned C ABI (mirrors rust/src/lib.rs)
rust/                   owned FFI shim crate + vendored crate closure
```

Qualification evidence, conformance matrix, and the decision record live in
`docs/adr/0024-*.md` and `docs/tokenizer-capabilities.md`.
