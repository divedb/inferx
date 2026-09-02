# Tensor and buffer contract

M2 provides the installed `inferx::tensor` CPU contract used by later CUDA
operators. ADR 0014 is normative; this document is the caller guide.

## Values and layouts

`Dtype` is a checked closed set from one-byte bool/integer storage through
FP16, BF16, FP32, and FP64. FP16/BF16 are storage-only in M2. Always use
`DtypeSize`, `DtypeName`, `IsIntegral`, and `IsFloatingPoint`; invalid enum
values return `InvalidArgument`.

`Shape::Create` and `Strides::CreateElements` accept rank 0 through 8 and keep
their values inline. A scalar has rank zero and one element. Any zero extent
makes an empty tensor. Strides are nonnegative element counts, never bytes;
zero stride is legal only for an extent of zero or one. Product, reach, and
byte conversions fail with `OutOfRange` before state is created.

`AnalyzeLayout` reports:

- row-major contiguity;
- dense versus padded storage;
- conservative non-overlap versus possible overlap; and
- maximum element/byte offset and total reachable bytes.

Immutable views may represent overlap. Mutable views require a proven
non-overlapping layout. Negative and broadcast strides are not M2 features.

## Allocation ownership

An `AllocationRequest` names device, memory kind, byte count, power-of-two
alignment, and accounting category. `CpuAllocator` accepts host memory.
CUDA allocators are build-only platform components; managed memory remains
`Unimplemented` in M2.

The default `CpuAllocator` is standalone. Constructing it with an
`AllocationAccounting&` (normally `MemoryTracker`) reserves before aligned
allocation, commits exactly one charge on success, and rolls back every
failure. The supplied accounting object must outlive buffers allocated through
that allocator; buffer allocation domains themselves still outlive an
allocator move or destruction.

`Buffer` uniquely owns one allocation and is move-only. `Release()` is the
normal, status-returning end of ownership. Its destructor is a no-throw safety
path whose behavior is chosen by the allocation domain: CPU memory is freed,
while CUDA abandonment records a leak and poisons the domain rather than
performing a potentially synchronizing free. Move leaves the source empty.
Zero-byte allocations retain metadata but have no address/deleter call.

`BufferView` and `MutableBufferView` are bounded and non-owning. They remain
valid across a move of their buffer, but not its release or destruction.
Subviews check `offset + size` and reduce their alignment guarantee. Host byte
spans are available only for host and pinned-host memory; device and managed
views reject host access. Allocation IDs, not pointer values, define alias
identity. Raw address access is confined to the non-installed CUDA platform
shim.

## Tensor views and copies

`TensorView::Create` validates shape/stride rank, dtype and offset alignment,
reachable bytes, and the backing range. `MutableTensorView::Create` adds the
non-overlap requirement. Empty views may end exactly at a buffer boundary.

`Slice(axis, begin, end, step)` is half-open with a positive step.
`Permute(axes)` requires a complete unique permutation. `Reshape(shape)` is
metadata-only and requires contiguous input with exactly the same element
count. None of these operations owns or copies storage.

`CopyTensorCpu` requires equal dtype/shape and host-addressable storage. It
walks logical row-major indices for arbitrary supported strides. Contiguous
copies use overlap-safe `memmove`; overlapping strided source/destination
ranges are rejected. `FillBufferCpu` is the explicit byte-fill primitive.

## Lifetime example

```cpp
inferx::CpuAllocator allocator;
auto buffer = allocator.Allocate({
    inferx::Device::Host(), inferx::MemoryKind::kHost,
    inferx::ByteCount(4096), inferx::ByteCount(64),
    inferx::MemoryCategory::kTest});
if (!buffer.ok()) return buffer.status();

auto shape = inferx::Shape::Create(std::array<uint64_t, 2>{16, 16});
auto strides = inferx::Strides::Contiguous(*shape);
auto bytes = buffer->MutableView({inferx::ByteCount(0),
                                  inferx::ByteCount(4096)});
auto tensor = inferx::MutableTensorView::Create(
    *bytes, inferx::Dtype::kFloat32, *shape, *strides);
if (!tensor.ok()) return tensor.status();

// All views must be finished before explicit release.
return buffer->Release();
```

Production code should use a semantic memory category rather than `kTest`.
