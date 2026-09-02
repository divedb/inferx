# communication kernels (scaffold)

Reserved category in the ADR 0031 kernels taxonomy. No operator contracts or
kernels live here yet; this directory exists so the taxonomy is stable while
contracts are added incrementally.

Planned provider chain (strongest first): hpc-ops -> flashinfer -> custom
CUTLASS -> owned. When an operator contract for this category lands in
`include/inferx/ops`, implementations register through
`kernels/cuda/dispatch` and become selectable through the unified
`inferx::kernels` dispatch surface without upper-layer changes.
