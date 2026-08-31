# ADR 0020: local artifact trust boundary

- Status: Accepted
- Date: 2026-08-31
- Owner: artifacts

## Context

Checkpoint filenames and shard-index entries are untrusted, while path canonicalization followed by
a normal `open` has a time-of-check/time-of-use race. Read-only mappings also do not make a mutable
inode safe from truncation.

## Decision

InferX accepts only an explicit local directory. The selected root may be a symlink because the
operator chose it; it is canonicalized once and opened as a directory fd. Every descendant is a
validated UTF-8 `SafeRelativePath` and is opened relative to that fd. Linux uses `openat2` with
`RESOLVE_BENEATH`, `RESOLVE_NO_SYMLINKS`, and `RESOLVE_NO_MAGICLINKS`; kernels without it use an
`openat` component walk with `O_NOFOLLOW`. Only regular files are accepted.

An opened file is identified by device, inode, type/mode, size, mtime, and ctime. Reads use `pread`;
identity is checked before and after parsing/hashing and again after reopening. A model directory
must be atomically staged and remain immutable while loaded. Malicious truncation of a live mapping,
which Linux may report as `SIGBUS`, is outside the in-process recovery guarantee.

## Alternatives

- `weakly_canonical` plus `open` was rejected because a descendant can change after validation.
- Allowing descendant symlinks was rejected because it breaks the root trust boundary.
- Signal-handler recovery from mapping faults was rejected as unsafe process control.

## Consequences

There is no recursive scan, canonical-path reopen, pickle, device file, FIFO, socket, remote lookup,
or symlink below the selected root. Root relocation does not affect identity. Deployment may add
fs-verity or a content-addressed store later without weakening this API.

## Validation evidence

`ModelLocatorTest.RejectsDescendantSymlink`, safe-path unit cases, file mutation checks, and mapping
budget lifetime tests exercise the implemented boundary. Both openat2 and fallback configurations
must remain in CI.

## Supersession

A later ADR may require fs-verity or an immutable content store; it may not reintroduce unchecked
path-based opens.
