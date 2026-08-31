# Qualification report: tokenizer (divedb/tokenizer)

- Manifest entry: `tokenizer` — candidate, feature `tokenization` (M3), owner `tokenization`
- Pin: `f109b7aef148dd4866a3dae7a8e5a6d221f95c75` (uninitialized in the core profile)
- License: MIT (`LICENSE` at the pin, verified via raw.githubusercontent.com)

## 1. Which InferX contract would use it?

Native `tokenizer.json` encode/decode behind the `Tokenizer` interface with incremental
detokenization (plan section 8.3).

## 2. Required now, deferred, experimental, or rejected?

Candidate, deferred to M3. M3 owns the decision.

## 3. Source and transitive dependencies

Small single-purpose C++ repository; closure audit pending (dependencies, build system,
any vendored copies) as part of the M3 spike.

## 4. Toolchain/C++23 compatibility

Unverified at the pin; M3 must build it with GCC 13/Clang 18 in C++23 host mode (or
isolate it at its own standard if unavoidable, per ADR 0005's isolation rule).

## 5. Runtime behavior caveats

Audit pending: thread safety of concurrent `Encode`/`Decode`, whether it throws on
malformed input (adapter must catch per ADR 0004), global state, and UTF-8 error
policy (plan section 8.3 requires defined invalid-UTF-8 behavior).

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

M3 qualification: initialize, wrap, run differential/fuzz suites; upgrades re-run them.

## 10. Disposition and approvals

**Candidate** (deferred), owner `tokenization`; correctness/thread-safety/license
approval owned by M3.
