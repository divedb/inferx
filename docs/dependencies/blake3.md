# BLAKE3 qualification

- Revision: `f3149ec5bb5449af877ba20377a11008ff499fa2` (official C 1.8.7)
- License: CC0-1.0 or Apache-2.0 or Apache-2.0 with LLVM exception
- Owner: artifacts
- Status: approved

InferX builds `blake3.c`, `blake3_dispatch.c`, and `blake3_portable.c`. SSE2, SSE4.1, AVX2, and
AVX-512 dispatch are disabled for the M3 baseline, and TBB/assembly are not linked. The C API is
private to `digest.cc`; public code sees only `Digest256` and `Hasher`.
Bundled installs export the archive as the implementation-only
`inferx::blake3_internal` target; no BLAKE3 header or source-tree include path
appears in the public artifact API.

Qualification includes official empty-input and `abc` vectors, incremental/one-shot equivalence,
lowercase hex round trips, finalization-state checks, complete-file hashing, range hashing, and file
mutation checks. SIMD may be enabled only after byte-equivalence and platform-dispatch testing.
