# Model artifact format and trust boundary

InferX opens an explicitly selected local directory and never resolves a Hub ID. The root is opened
once; all fixed artifacts and index/manifest shard names pass `SafeRelativePath` and are opened below
the directory fd. Valid paths are nonempty UTF-8 slash paths without absolute prefixes, NUL,
backslash, empty, dot, or dot-dot components. Descendant symlinks and non-regular files fail.

Every opened file records device, inode, mode, size, mtime, and ctime. Reads use exact `pread` loops;
hash and parse operations recheck identity. Mapping accepts only a validated byte range, aligns the
private read-only map to host pages, exposes only requested bytes, and transactionally accounts the
rounded bytes and active slot. Zero-byte tensors do not call `mmap`.
Ranges larger than one mapping window are consumed with `MappedTensorReader`. The reader owns a
duplicate of the already-rooted file descriptor, shares the originating pool's count/byte budget,
and advances only after a window is mapped successfully. Returned windows are contiguous and never
cross the validated requested range.

The optional `inferx.manifest.json` schema is version 1 and contains `model_revision`,
`weights_entry`, and unique file entries with path, uint64 size, and lowercase BLAKE3-256. All
consumed files must be declared and all declarations consumed. The manifest is not self-listed.

Safetensors validation is ordered: eight-byte length, bounded header, UTF-8 JSON object, duplicate
keys, exact tensor fields, known wire dtype, bounded shape, checked bit-packed byte count, offsets,
file bounds, and sorted contiguous payload coverage. The header can end only with ASCII-space
padding. Shard indices require a nonempty complete `weight_map`; each shard tensor and index entry
must correspond exactly, and optional `metadata.total_size` counts tensor payload bytes.

Errors distinguish invalid operator input (`InvalidArgument`), missing files (`NotFound`), resource
caps (`ResourceExhausted`), format corruption/integrity mismatch (`DataLoss`), concurrent mutation
(`Aborted`), and valid but unsupported capability (`Unimplemented`).
