# Qualification report: tokenizer (owned adaptation of divedb/tokenizer)

- Manifest entries: `tokenizer` (owned vendor package), `nlohmann-json`, `minja` —
  feature `tokenization` (M3), owner `tokenization`
- Provenance: divedb/tokenizer @ `f109b7aef148dd4866a3dae7a8e5a6d221f95c75` (MIT),
  adapted locally; the Rust FFI shim and its build are new InferX code
- Decision: **Accepted (ADR 0024, 2026-09-02)**

## 1. Which InferX contract uses it?

Native `tokenizer.json` encode/decode/streaming-decode/chat-template behind
`inferx::tokenization::Tokenizer` (m3.md section 13), plus prompt processing
and the validated model package (m3.md section 14).

## 2. Required/deferred/experimental/rejected?

Accepted, replacing the 2026-08-31 rejection of the unchanged upstream. The
rejection's four findings are each closed:

| Rejection finding | Closure |
|---|---|
| force-added duplicate deps; nested-tree symlink mutation | owned vendor CMake; parent-provided Abseil; no add_subdirectory of upstream; no symlinks |
| unconditional Hub/curl/OpenSSL network closure | hub/cache/curl/OpenSSL code deleted; loading consumes bytes only |
| `unwrap()`/abort reachable from malformed bytes | owned shim: every entry `catch_unwind`-guarded, status + owned error buffer; subprocess fuzz corpus proves no abort |
| no upstream streaming decode state | `ixtok_stream_new/step/finish` externalize the upstream `step_decode_stream` state |

## 3. Source and transitive dependencies

- Rust engine: official Hugging Face `tokenizers` crate `=0.21.2`; the full
  crates.io closure (80 crates) is vendored under
  `third_party/tokenizer/rust/vendor/` and pinned by `Cargo.lock`. Licenses:
  Apache-2.0/MIT (tokenizers, rayon, serde, ...), BSD-2-Clause (onig_sys's
  vendored Oniguruma), Unicode-3.0 (unicode-ident). The build is offline by
  configuration (`[net] offline = true`).
- Chat-template closure: `minja` (Apache-2.0) and `nlohmann/json` (MIT) as
  pinned submodules; private to the vendor target, never in an InferX
  public header.
- C++ layer deps: parent Abseil only.

## 4. Toolchain compatibility

C++23 (GCC 13 / Clang 18 lanes) via the adapted divedb sources; Rust via
`cargo` (1.97-era toolchain tested) building a `staticlib` with `panic =
"unwind"` so `catch_unwind` can convert engine panics into FFI error
returns. The staticlib links `Threads`/`dl` only.

## 5. Runtime behavior caveats

One thread-affine engine handle per exclusive instance; the InferX facade
lends instances under a mutex/condvar and the pool owns one per worker.
Engine-internal Rayon parallelism is disabled on the serving path. Decode
and stream results are caller-owned buffers; nothing aliases handle state.

## 6. API stability

Unstable, first-party, and completely hidden behind
`inferx::tokenization`/`inferx::input`; no vendor type appears in any InferX
public header. Upgrades require re-running the differential corpus.

## 7. License/security

MIT (adapted C++), Apache-2.0/MIT (Rust engine closure), Apache-2.0 (minja),
MIT (nlohmann). The parser is fuzzed in a subprocess (committed corpus plus
seeded mutations); malformed input is a typed error, never an abort. No
credentials, network, or paths outside the rooted artifact session.

## 8. Cost

Offline cargo build (~12 s warm, ~1 min cold) producing a ~37 MB static
archive; per-instance engine construction ~0.3 s per real checkpoint
(Qwen2.5 measured on the dev host).

## 9. Upgrade/rollback

The vendor tree is repo-owned: upgrades are repo commits diffing against the
recorded upstream revision, followed by the full differential corpus,
subprocess corpus, and TSan pool gate. Rollback is `git revert`.

## 10. Disposition

Approved (ADR 0024). `INFERX_ENABLE_TOKENIZATION` defaults ON; missing
cargo/submodules fail configure with the exact bootstrap remediation.
