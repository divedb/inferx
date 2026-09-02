# ADR 0024: tokenizer backend and concurrency

- Status: Accepted
- Date: 2026-08-31 (proposed); 2026-09-02 accepted
- Owner: tokenization

## Context and audit result

The removed `divedb/tokenizer` revision was
`f109b7aef148dd4866a3dae7a8e5a6d221f95c75`. Its logical ownership model is suitable—one
thread-affine handle per worker—but its build unconditionally adds duplicate dependencies and a
network/OpenSSL closure. Its Rust shim has `unwrap()` paths reachable from malformed bytes and it
does not expose upstream streaming decode state.

## Decision

The replacement is an **owned local-only adaptation** of that revision, vendored at
`third_party/tokenizer/` and qualified against every criterion this ADR originally demanded:

1. the C++ half is the audited divedb/tokenizer code (MIT; `LICENSE.divedb-tokenizer`),
   adapted to consume checkpoint **bytes** (`LocalArtifacts`) instead of discovering files:
   Hub/cache/curl/OpenSSL, directory scanning, the SentencePiece backend, batch encode, and
   runtime `AddTokens` are removed;
2. the Rust half is a **new owned shim** (`rust/src/lib.rs`, crate `inferx-tokenizers-c`) over
   the official Hugging Face `tokenizers` engine, pinned at `=0.21.2` with the full crates.io
   closure vendored under `rust/vendor/` (offline builds only, `rust/.cargo/config.toml`
   forbids network). Every entry point is `catch_unwind`-guarded and returns a status code plus
   an owned error buffer; construction from malformed `tokenizer.json` is `DataLoss`, invalid
   UTF-8 input is `InvalidArgument`, and **no panic, abort, or unwrap crosses the FFI**;
3. every returned buffer is caller-owned (`ixtok_free_bytes`/`ixtok_free_ids`); no returned
   pointer aliases handle-owned scratch;
4. streaming decode state is **externalized**: `ixtok_stream_new/step/finish` hold exactly the
   upstream `step_decode_stream` state (pending ids, committed prefix, prefix index) and borrow
   a tokenizer only per call, so any pool instance can serve any decoder;
5. engine-internal parallelism is set explicitly (`ixtok_set_parallelism(false)` on the serving
   path): InferX workers own exclusive handles and encode single strings, so the engine's Rayon
   pool must not oversubscribe them;
6. parent-provided Abseil only; no dependency is added, force-set, or symlinked. nlohmann/json
   and minja (the chat-template closure, chat rendering retained per the original audit) are
   pinned git submodules at `third_party/nlohmann-json` and `third_party/minja`;
7. complete license records for the recursive closure exist in `docs/dependencies/tokenizer.md`,
   `docs/dependencies/nlohmann-json.md`, and `docs/dependencies/minja.md`; the Rust crate
   closure (80 crates, all Apache-2.0/MIT/BSD/Unicode) is recorded in
   `third_party/tokenizer/rust/Cargo.lock` and summarized in the tokenizer record.

Provenance note: the exact upstream pin `f109b7ae` was re-fetched for the adaptation; the
original audit remains in force for everything not listed as changed above. The upstream
tokenizers-cpp submodule pin (`586b0ee1`) is no longer fetchable upstream; nothing of it is
vendored—the owned shim replaces it entirely.

## Qualification evidence (2026-09-02, RTX 4080 SUPER host, GCC 13 / clang 18)

- **Exact 10,000-case Hugging Face differential** (`inferx_tokenizer_differential_test`):
  4 checkpoints × 2,500 deterministic cases — Qwen2.5 (BPE + NFC + ByteLevel),
  TinyLlama (BPE + byte-fallback decoder), bert-base-uncased (WordPiece), t5-small (Unigram +
  Precompiled) — encode with and without special tokens, one-shot decode, and streaming decode
  equality (`chunks ⊕ Finish == one-shot`) all byte-exact against the pinned Python oracle
  `tokenizers==0.21.1`. Corpus: `tests/fixtures/model/tokenizer_reference/`, regenerated
  deterministically by `tools/fixtures/generate_tokenizer_reference.py`.
- **Malformed-input subprocess corpus** (`inferx_tokenizer_subprocess_test`): 30 structured
  hostile blobs (wrong-typed components, unknown component types, negative/huge ids, deep
  nesting) plus 150 seeded byte-level mutations of a real 128 KiB tokenizer.json; every child
  exits cleanly, none dies by signal.
- **TSan pool stress** (`inferx_tokenizer_pool_stress_test`, `tsan` preset, clang): 8
  concurrent submitters, 400 jobs, exactly-one completion per accepted job, parallel streaming
  decoders borrowing the shared instance set, byte budget returning to zero — no ThreadSanitizer
  reports.
- The unit smoke suite (`inferx_tokenization_test`) covers round-trip, streaming equality,
  and error-not-abort behavior on a real checkpoint.

## Consequences

M3's tokenizer gate is closed: `ValidatedModelPackage` publishes a qualified tokenizer with
metadata, the model/tokenizer cross-check, and the fingerprint `tokenizer_capability` record;
`inferx-model-inspect` reports `tokenizer_qualified=true` and the real BLAKE3 fingerprint.
`INFERX_ENABLE_TOKENIZATION` defaults ON and requires `cargo` plus the two submodules; the
configure error carries the exact remediation.

The `inferx::tokenization` and `inferx::input` targets remain **non-installed**: their static
link closure contains the Rust engine archive, and packaging that closure (installed target or
system provider) is a future packaging ADR. The independent install consumer continues to cover
artifacts/model.

Supported-component conformance is named in `docs/tokenizer-capabilities.md`.

## Alternatives

- Embedding the original CMake project unchanged was rejected for duplicate targets, network
  closure, filesystem mutation, and abort behavior.
- Catching C++ exceptions was rejected because Rust abort/panic paths do not become recoverable.
- A home-grown tokenizer or repeated-prefix incremental decode was rejected for fidelity and
  already-emitted-byte correctness.

## Supersession

None. This ADR is accepted on the evidence above; the backend may be replaced only through a
superseding ADR that satisfies the same public contract and corpus.
