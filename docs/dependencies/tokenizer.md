# Qualification report: tokenizer (divedb/tokenizer)

- Manifest entry: `tokenizer` — candidate, feature `tokenization` (M3), owner `tokenization`
- Pin: `f109b7aef148dd4866a3dae7a8e5a6d221f95c75` (audited 2026-08-31)
- License: MIT (`LICENSE` at the pin, verified via raw.githubusercontent.com)

## 1. Which InferX contract would use it?

Native `tokenizer.json` encode/decode behind the `Tokenizer` interface with incremental
detokenization (plan section 8.3).

## 2. Required now, deferred, experimental, or rejected?

Rejected unchanged by M3.0. The gitlink remains as audit evidence and is not configured or linked.

## 3. Source and transitive dependencies

The pin force-configures private Abseil and GoogleTest copies, nlohmann/json, minja,
tokenizers-cpp/SentencePiece, curl, and system OpenSSL. Its default target always compiles Hub/cache
and HTTP sources. It mutates a nested SentencePiece tree with a symlink at configure time and uses
`CACHE ... FORCE`, so it cannot be embedded under InferX's dependency policy unchanged.

## 4. Toolchain/C++23 compatibility

The surface is C++20-compatible, but the unchanged CMake composition collides with InferX's Abseil
targets before a qualified C++23 integration can be produced.

## 5. Runtime behavior caveats

`PretrainedTokenizer` is explicitly thread-affine and non-thread-safe. Rust encode/decode results
alias handle-owned scratch. More importantly, the C shim calls Rust `unwrap()` for malformed
`tokenizer.json` and invalid UTF-8; the C++ layer can pre-screen common cases but cannot guarantee
that arbitrary malformed tokenizer structures will not abort. The public surface has no upstream
streaming decode state.

## 6. API stability and namespaces

Small project (single-digit stars); API stability must be treated as unstable — the M3
adapter wraps it entirely behind InferX's `Tokenizer` interface, and differential tests
against Hugging Face Tokenizers are the acceptance gate.

## 7. License/notice obligations and security

MIT at the pinned revision. Fuzz the parser on untrusted `tokenizer.json` (plan section
17.3 requires the security audit). No known advisories; small attack surface but
unaudited.

## 8. Binary/build/startup cost

Expected small; measure at M3.

## 9. Upgrade/rollback procedure

An upgrade is eligible only when it supplies local-only composition, parent-provided Abseil/tests,
owned error returns across FFI, and upstream streaming decode state. It then runs the 10,000-case
differential corpus, subprocess malformed-input corpus, TSan pool stress, and license/SBOM audit.

## 10. Disposition and approvals

**Rejected unchanged**, owner `tokenization`. `INFERX_ENABLE_TOKENIZATION=ON` fails configuration
until ADR 0025 names an approved replacement pin. The rejection is a hard M3 completion gate, not a
runtime fallback.
