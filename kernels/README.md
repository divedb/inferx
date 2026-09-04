# InferX kernel layer

This directory contains InferX-owned backend-neutral kernel contracts and
backend implementations tracked in `../kernels.md`. TokenSpeed is the audited
workload and compatibility source, not the owner of the public C++ namespace.
The directory can be included from a parent CMake project or configured
directly while the surrounding runtime is rebuilt.

InferX-owned CUDA device kernels use the CuTe tensor and layout abstractions
from the pinned `third_party/cutlass` submodule. Backend-neutral headers do not
expose CUTLASS types, and CUDA runtime/cuBLAS operations remain native library
calls rather than wrapper kernels.

## K66 memory primitives

Include `inferx/kernels/memory.h` and call:

- `ZeroAsync` for bytewise zeroing;
- `FillBytesAsync` for a repeated byte;
- `FillAsync` for a repeated 1/2/4/8-byte typed storage pattern;
- `CopyAsync` for a same-backend device-to-device copy.

The API is allocation-free and asynchronous on the opaque caller stream. It
accepts exact subranges, treats zero-sized requests and exact-alias copies as
no-ops, and rejects null nonempty ranges, malformed typed ranges, size
mismatch, and partial overlap (overlap checks are overflow-safe). CPU/device,
peer-device, and cache-range transfers are intentionally outside this
contract and belong to K63. Full reference below.

## K09 layout operations

Include `inferx/kernels/layout.h`. Views are free (`TransposeView`,
`SliceView`, `ReshapeView`, `MakeContiguousView`); copies are device work
(`CopyLayoutAsync` between arbitrary stride layouts, `ConcatAsync` along one
dimension). Copies between two contiguous views ride the K66 block-copy
primitives; strided copies launch statically rank-specialized CuTe layouts on
CUDA and a grid-stride HIP kernel on ROCm, and are rejected with `kUnsupported`
on Ascend until an AscendC kernel lands.

## Stage-1 P0 operations (CUDA only)

The Stage-1 families below are implemented for CUDA only; other backends
report `kUnavailable` (and `kCPU` reports `kUnsupported`) until they gain
implementations. Every operation is enqueued on the caller stream,
allocates nothing per call, and validates before dispatch (null pointers,
dtype restrictions, conservative overlap rejection, geometry checks).

| Family | Header | Operations |
|---|---|---|
| K01 embedding | `embedding.h` | `EmbeddingGatherAsync` — row gather with out-of-shard IDs zeroed |
| K03 dense GEMM | `gemm.h` | `DenseGemmAsync` — `[M,K] x [N,K]^T` via cuBLAS, FP32 accumulate |
| K10 gather/scatter | `scatter.h` | `GatherRowsAsync`, `ScatterRowsAsync` (duplicate winner unspecified) |
| K11 scans | `scan.h` | `ExclusiveSumAsync`, `InclusiveSumAsync` — exact int32/int64 |
| K12/K65 metadata | `metadata.h` | `BuildCacheLocationsAsync` (K11 offsets + `-1` sentinel), `BuildSequenceIndexAsync` (gap-aware), `LogScalingTauAsync` |
| K13 norm | `norm.h` | `RmsnormAsync`, `AddRmsnormAsync` (fused residual, FP32 stats) |
| K17/K19 activation | `activation.h` | `SiluAndMulAsync`, `ResidualAddAsync` |
| K20 RoPE | `rope.h` | `ApplyRopeAsync` — half-rotary/interleaved, in-place or out-of-place |

Notes and honest caveats:

- cuBLAS is **dynamic-loaded on the first GEMM, after the CUDA runtime is
  forced to initialize**: linking `libcublas` directly makes `libcublasLt`'s
  ELF initializers run before CUDA init, which breaks device enumeration on
  WSL2 with the CUDA 12.0 toolkit. The lazy handle also installs a fixed
  workspace so graph capture never observes internal cuBLAS allocations.
- Index bounds for gather/scatter and RoPE positions are device-side caller
  contracts (checking them would require a host round-trip).
- The scan is a single-block chained-tile kernel: exact and deterministic,
  sized for batch metadata (thousands of entries); a multi-block variant is
  future work.
- Operation labels for `Status::Operation()` follow `family.op`, e.g.
  `scatter.gather`, `scan.exclusive_sum`, `metadata.cache_locations`,
  `gemm.dense`, `norm.add_rmsnorm`, `activation.silu_and_mul`, `rope.apply`.

## API reference

Public headers live in `include/inferx/kernels/`; every symbol is in
`inferx::kernels`.

### `inferx/kernels/status.h`

Every operation returns `Status` by value. `Status` is allocation-free and
`noexcept` — operation names and messages are copied into fixed storage
(truncated at 48 and 128 bytes respectively) — so results can be constructed
and propagated on any path, including `noexcept` backend entry points.

A default-constructed `Status` is success. Failures are built with the static
factories `Status::InvalidArgument/Unavailable/Unsupported(operation,
message)` and `Status::ResourceExhausted/BackendError(operation, message,
native_code)`. Inspect results with `IsOk()`, `Code()`, `Operation()`,
`Message()`, and `NativeCode()` (the raw `cudaError_t`/`hipError_t`/
`aclError` value as `int32_t`, or 0 when the backend supplies none).

`StatusCode` meaning and provenance:

| Code | Meaning | Produced by |
|---|---|---|
| `kOk` | Success | every layer |
| `kInvalidArgument` | Caller contract violation caught before dispatch: null nonempty range, typed size not a multiple of the pattern width, misaligned typed destination, copy size mismatch, or partial overlap | common validation |
| `kUnavailable` | Backend is not part of this build (or has no usable runtime); query `IsBackendCompiled` to distinguish | common dispatch |
| `kUnsupported` | Compiled backend rejects this operation tuple by capability, e.g. a non-repeated typed fill on Ascend (no AscendC pattern kernel yet), or any device-memory operation on `kCPU` | backend or common dispatch |
| `kResourceExhausted` | Backend resource failure such as an out-of-memory or launch-resource error | backend error mapping |
| `kBackendError` | Any other native failure; `NativeCode()` carries the raw error value | backend error mapping |

Validation failures identify the call through the `Operation()` labels
`memory.zero`, `memory.fill_bytes`, `memory.fill`, and `memory.copy`.

### `inferx/kernels/layout.h`

Types:

- `DType` — element types for layout moves (`kUint8`, `kBFloat16`,
  `kFloat16`, `kFloat32`, `kFloat64`, `kInt32`, `kInt64`); moves never
  convert, so the type only fixes `ElementSizeOf(dtype)` bytes.
- `TensorView` — a non-owning pointer + dtype + rank + shape + strides
  (strides in **elements**, rank at most `kMaxTensorRank` = 5; size-1
  dimensions carry arbitrary strides).

Host-side view transforms (metadata only, never touch device memory, return
`kInvalidArgument` for structural violations):

- `TransposeView(tensor, permutation, result)` — permutes dimensions.
- `SliceView(tensor, dim, offset, length, result)` — sub-view along one
  dimension; also covers splitting packed tensors.
- `ReshapeView(tensor, new_shape, result)` — a view only when the source is
  C-contiguous (`IsContiguous`); otherwise `kUnsupported`, and the caller
  copies first.
- `MakeContiguousView(like, data, result)` — builds a row-major destination
  sized like `like`; size it with `ElementCount`.
- `ElementCount(tensor, count)` — overflow-checked element count.

Device operations (async on the caller stream, allocation-free):

- `CopyLayoutAsync(source, destination, context)` — elementwise copy between
  arbitrary stride layouts with the same shape/dtype; exact aliases are
  no-ops, overlapping byte extents are rejected conservatively (same-
  allocation views whose boxes overlap are rejected even when their elements
  do not), and misaligned data pointers are rejected. Contiguous-to-
  contiguous copies ride the K66 block-copy primitives.
- `ConcatAsync(sources, count, dim, destination, context)` — concatenates
  along one dimension as per-source `CopyLayoutAsync` calls; sources may have
  differing strides.

Backend realization:

| Operation | CUDA | ROCm | Ascend |
|---|---|---|---|
| Copy, both contiguous | `cudaMemcpyAsync` | `hipMemcpyAsync` | `aclrtMemcpyAsync` |
| Copy, strided | rank-specialized CuTe layout kernel | HIP grid-stride kernel | `kUnsupported` until an AscendC kernel lands |
| Concat | per-source copies | per-source copies | contiguous-only tuples |

```cpp
#include "inferx/kernels/layout.h"

using inferx::kernels::DType;
using inferx::kernels::ExecutionContext;
using inferx::kernels::TensorView;
using inferx::kernels::CopyLayoutAsync;
using inferx::kernels::MakeContiguousView;
using inferx::kernels::TransposeView;

// Permute [T, H, D] heads-first and pack into a contiguous buffer.
TensorView transposed;
const uint32_t permutation[3] = {1, 0, 2};
if (!TransposeView(activations, permutation, &transposed).IsOk()) { /* ... */ }
TensorView packed;
if (!MakeContiguousView(transposed, scratch, &packed).IsOk()) { /* ... */ }
if (!CopyLayoutAsync(transposed, packed, context).IsOk()) { /* ... */ }
```

Note that `std::span` parameters (permutations, shapes) do not bind braced
initializer lists directly; materialize an array first as above.

### `inferx/kernels/memory.h`

Types:

- `DeviceBytes` / `MutableDeviceBytes` — a non-owning `{data, size}` byte
  range with a `uint64_t` size. Operations accept exact subranges.
- `FillPattern` — a repeated typed storage pattern. `FillPattern::From(value)`
  accepts integral or floating values of 1/2/4/8 bytes and is defined by the
  value's *storage bits*, not by numeric equivalence across types;
  `FillPattern::FromStorageBits16(bits)` builds an explicit 16-bit pattern
  (FP16/BF16 payloads such as `0x3c00`). `Bits()` and `Width()` expose the
  representation.

Operations — all enqueue on `context.native_stream`, allocate nothing, and
never synchronize. A zero-sized range is a successful no-op and may pass a
null pointer:

- `ZeroAsync(destination, context)` — bytewise zero fill.
- `FillBytesAsync(destination, value, context)` — repeat one byte.
- `FillAsync(destination, pattern, context)` — repeat a typed pattern; the
  size must be a multiple of the pattern width and the address aligned to it.
- `CopyAsync(source, destination, context)` — same-backend device-to-device
  copy; sizes must match, exact aliases are no-ops, and partial overlap is
  rejected. Host/device and peer-device transfers belong to K63.

The context (`inferx/kernels/context.h`) selects the backend and the native
stream: `Stream` wraps an opaque handle (`cudaStream_t`, `hipStream_t`, or
`aclrtStream`) — construct it with `Stream(handle)` and unwrap with
`Native()`; backends receive the raw handle. Which backends this build
contains is a compile-time fact: CMake generates
`inferx/kernels/build_config.h` from the `INFERX_ENABLE_*` options with
`inline constexpr` flags `kHasCuda`, `kHasRocm`, and `kHasAscend`.
`IsBackendCompiled(backend)` (itself `constexpr`) maps a `Backend` to those
flags; operations on an uncompiled backend return `kUnavailable`. `kCPU`
needs no toolchain and is always compiled in, but device-memory operations
reject it with `kUnsupported`.

Backend realization:

| Operation | CUDA | ROCm | Ascend |
|---|---|---|---|
| Zero / byte fill | `cudaMemsetAsync` | `hipMemsetAsync` | `aclrtMemsetAsync` |
| Typed fill, repeated byte | memset fast path | memset fast path | memset fast path |
| Typed fill, other patterns | CuTe tensor kernel | HIP grid-stride kernel | `kUnsupported` until an AscendC kernel lands |
| Device copy | `cudaMemcpyAsync` (D2D) | `hipMemcpyAsync` (D2D) | `aclrtMemcpyAsync` (D2D) |

CUDA and ROCm operations use stream-capture-capable APIs (graph capture and
replay verified on-device for CUDA); AscendCL has no equivalent capture
path.

Example:

```cpp
#include "inferx/kernels/memory.h"

using inferx::kernels::Backend;
using inferx::kernels::ExecutionContext;
using inferx::kernels::FillAsync;
using inferx::kernels::FillPattern;
using inferx::kernels::MutableDeviceBytes;
using inferx::kernels::Stream;

const ExecutionContext context{Backend::kCuda, Stream{device_stream}};
MutableDeviceBytes hidden{device_buffer, hidden_bytes};

if (const auto status = FillAsync(hidden, FillPattern::From<float>(0.0F), context);
    !status.IsOk()) {
  // Handle via status.Code(), status.Operation(), status.Message(),
  // and status.NativeCode().
}
```

## Building and tests

```sh
cmake -S kernels -B out/kernels-cpu -G Ninja -DBUILD_TESTING=ON
cmake --build out/kernels-cpu
ctest --test-dir out/kernels-cpu --output-on-failure

cmake -S kernels -B out/kernels-cuda -G Ninja \
  -DBUILD_TESTING=ON -DINFERX_ENABLE_CUDA=ON
cmake --build out/kernels-cuda
ctest --test-dir out/kernels-cuda --output-on-failure

cmake -S kernels -B out/kernels-ascend -G Ninja \
  -DBUILD_TESTING=ON -DINFERX_ENABLE_ASCEND=ON \
  -DINFERX_ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
cmake --build out/kernels-ascend
ctest --test-dir out/kernels-ascend --output-on-failure
```

One device backend (CUDA, ROCm, or Ascend) may be enabled per build; the
options are an explicit request, never auto-detected, and
`find_package(... REQUIRED)` validates the toolchain when a backend is
requested. The CUDA build defaults to `CMAKE_CUDA_ARCHITECTURES=75`; pass
`-DCMAKE_CUDA_ARCHITECTURES=<list>` for tuned SASS. The Ascend build locates
CANN through `INFERX_ASCEND_CANN_PATH`, `ASCEND_TOOLKIT_HOME`,
`ASCEND_HOME_PATH`, or `/usr/local/Ascend/ascend-toolkit/latest`. The
generated `build_config.h` lands in the build tree's `include/` and mirrors
the enabled backends as `kHas*` flags.

Initialize `third_party/cutlass` before enabling CUDA. CMake checks for the
pinned CuTe headers and fails configuration with an actionable error when the
submodule is absent.
