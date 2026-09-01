# ADR 0014: Tensor and buffer contract

- Status: Accepted
- Date: 2026-08-31
- Deciding authority: `docs/milestones/m2.md` sections 4 and 7

## Context

Every later operator and scheduler needs one hardware-neutral description of
tensor layout and allocation lifetime. Hidden byte strides, unchecked shape
arithmetic, pointer identity, and mutable overlapping views would make the CPU
oracle and CUDA path disagree in ways that cannot be contained at a module
boundary.

## Decision

1. `DType` is a closed checked enum; FP16/BF16 are storage types and bool is
   one byte. Generic tensor headers contain no CUDA types.
2. Shape and element-stride values use inline rank-eight storage. Rank zero is
   a scalar, any zero extent is empty, and all reach/product calculations are
   checked. Negative strides and implicit broadcasting are outside M2.
3. `LayoutAnalysis` reports row-major contiguity, density, reachable bytes,
   and conservative overlap. Immutable views may overlap; mutable views may
   not. Slice is half-open with a positive step, permutation is complete and
   unique, and reshape requires contiguous equal-element-count input.
4. `Buffer` is move-only and uniquely owns an allocation through an explicit
   `AllocationDomain`. `Release()` reports errors; the no-throw destructor uses
   the domain's abandonment policy. Views are bounded, non-owning, and expose
   host spans only for host-addressable memory.
5. Allocation identity is a process-unique `AllocationId`, never an address.
   Alignment and every subview range are validated with checked arithmetic.
6. `CopyTensorCpu` is the independent logical-row-major oracle: contiguous
   copies use `memmove`, strided overlap is rejected, and no dtype conversion
   occurs.

## Alternatives

- Raw pointers as public identity were rejected because allocators reuse
  addresses and device pointers cannot be safely inspected by generic code.
- Byte strides were rejected because element strides preserve the same layout
  across dtype-width checks and match kernel indexing.
- Negative/broadcast strides were rejected because writable aliasing and
  reverse-bound proofs require an explicit later contract.
- Shared allocation ownership was rejected. Only the deallocation domain is
  shared so an allocator facade may die before its buffers.
- Implicit copies or dtype conversion during view transformations were
  rejected; transformations remain allocation-free metadata operations.

## Consequences

The API rejects layouts it cannot prove safe and requires callers to retain
the owning `Buffer`. Future broadcast, negative-stride, quantized, or owning
tensor types must extend this descriptor explicitly. Changing stride units,
rank, or allocation identity is a migration-visible API change.

## Validation evidence

`inferx_m2_tensor_test`, the 10,000-case CPU transformation stress test, the
installed consumer, header checks, and ASan/UBSan lanes validate this record.
CUDA correctness uses the CPU copy as its byte-for-byte oracle.

## Supersession

A successor must preserve checked reach and explicit ownership while providing
a migration for every public tensor/view constructor.
