# Tokenizer capability status

No production tokenizer capability is currently enabled. The audited divedb/tokenizer pin
`f109b7aef148dd4866a3dae7a8e5a6d221f95c75` is rejected unchanged for four independent reasons:

- its embedded build force-adds duplicate Abseil/test dependencies and mutates a nested source tree;
- Hub/curl/OpenSSL code is unconditional rather than absent from a local-only target;
- malformed tokenizer bytes can reach Rust `unwrap()`/abort paths; and
- upstream incremental decode state is not exposed by its C++/C surface.

`INFERX_ENABLE_TOKENIZATION=ON` therefore fails configure. There is no fallback BPE, Unigram,
SentencePiece, Unicode normalizer, byte decoder, template approximation, or repeated-prefix streaming
implementation in InferX.

An approvable replacement must support local `tokenizer.json`, owned status/error buffers, valid
UTF-8 enforcement, model-range IDs, serialized added tokens and accepted pipeline components,
checkpoint-declared chat templates, and upstream `step_decode_stream` semantics with cleanup off.
It must declare exact engine/component versions and pass the M3 10,000-case differential corpus,
malformed-input subprocess fuzzing, exclusive-instance pool stress, and TSan. This document becomes a
component/type matrix when ADR 0024 names that revision.
