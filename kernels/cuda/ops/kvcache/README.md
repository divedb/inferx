# KV-cache kernels

Standalone cache-maintenance kernels (append/compaction). At the pinned
revisions neither hpc-ops nor flashinfer exposes a standalone contiguous-BSHD
append that matches the InferX operator contracts (hpc-ops fuses rope+norm+KV
store in `src/rope`; flashinfer's append is fused into paged RoPE kernels), so
no custom kernel is carried here either: cache writes are performed by the
attention provider that owns the launch sequence, as in the retired platform layer.
