# Attention kernels

Provider status at the pinned revisions (ADR 0032):

- **hpc-ops** (`third_party/hpc-ops`): first preference. Attention kernels at
  the pin are SM90/SM100/SM103 warp-specialized builds behind the project's
  own one-arch-per-module CMake; unavailable on the SM89 runner and not yet
  qualified for InferX contracts.
- **flashinfer** (`third_party/flashinfer`): decode/prefill device templates
  exist (`attention/decode.cuh`, `attention/prefill.cuh`) but are paged-KV
  based; mapping InferX's contiguous BSHD cache onto `paged_kv_t` plus the
  scheduler params is a pending qualification item (ADR 0032 gap log).

Until an adapter qualifies, attention execution stays on the M4 platform
layer (`platform/cuda/src/ops/cuda_attention.cc`), which this architecture
leaves untouched.
