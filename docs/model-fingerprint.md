# Model fingerprint schema version 1

`Digest256` and `Hasher` wrap official portable BLAKE3 C 1.8.7. A digest is exactly 32 bytes; text is
64 lowercase hex characters. File hashing streams from the opened fd and includes complete file
bytes, not only a safetensors header or normalized JSON.

`ModelFingerprint::Build` starts with `inferx.model-fingerprint` and little-endian schema version 1.
It appends length-delimited records for architecture, canonical semantic config, model and weight
plan schemas, qualified tokenizer capability, each path-sorted artifact `(safe path, uint64 size,
digest)`, manifest revision, source layout, quantization, adapter, and RoPE policy. Duplicate artifact
paths fail. Numeric canonicalization is fixed-width little-endian; finite floating fields enter as
normalized IEEE-754 binary64 bits through the semantic model record.

Absolute root, inode, timestamps, input JSON ordering, and unrelated process flags are excluded.
Raw config/tokenizer/weight differences still change their artifact digests. A base fingerprint is
not emitted until qualified tokenizer metadata is present; callers must not substitute the empty
capability record and call it a model identity.
