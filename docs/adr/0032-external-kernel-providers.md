# ADR 0032: external kernel provider qualification at the pinned revisions

- Status: Accepted (provisional; evidence recorded below)
- Date: 2026-09-01
- Owner: kernels layer
- Supersedes: ADR 0029 (for the kernels-architecture provider chain only; the
  M4 platform-layer FlashInfer rejection stands unchanged)

## Context

ADR 0031 fixes the provider preference chain hpc-ops → flashinfer → CUTLASS
custom → vendor/owned. Reuse is only real if the providers are actually
integrated at the recorded pins. This ADR records what was integrated, what
was audited and found non-matching, and what remains open. All evidence below
comes from the pinned revisions in `third_party/manifest.json`:

- cutlass `dc45f979ae336a235da1676b311f35efeb30149a` (CUTLASS 4.8)
- flashinfer `44428003ba219c14b5473fefec8f7bfd4b72178e`
  (+ its recorded nested closure: `3rdparty/cccl @ 16bd510c9b712e82b0ab6cbb630d8e29ba1f7116`)
- hpc-ops `f39028d9f5ab77f71906fbf929d1b611859ab6b7`

## Decision

The three options `INFERX_ENABLE_FLASHINFER`, `INFERX_ENABLE_CUTLASS`, and
`INFERX_ENABLE_HPC_OPS` replace the M4 fatal guards: flashinfer and cutlass
gate real ahead-of-time integration; hpc-ops reserves its first-position
chain slot. The `cuda-kernels` preset enables all three. Integration uses
only ahead-of-time compilation of the pinned sources; no JIT kernel cache, no
runtime cubin loading, no Python in the link closure.

## Per-operator audit and integration state

| operator | hpc-ops at pin | flashinfer at pin | integrated provider |
|---|---|---|---|
| rms_norm | `fused_rmsnorm_with_scale.h` includes ATen/PyTorch headers — blocked | `norm.cuh RMSNormKernel` (warp-reduced, vectorized) | **flashinfer (integrated, oracle-verified)** |
| rope | fused qk-norm+rope+KV-store, BF16-only, cos/sin table — contract mismatch | `pos_enc.cuh BatchQKApplyRotaryPosIdsKernel` (NeoX non-interleave, per-token pos ids) | **flashinfer (integrated, oracle-verified; head dims {32,64,128,256})** |
| gemm / logits | group GEMM is FP8-only with SM90 TMA descriptor workspaces — contract mismatch | CUTLASS runner is BF16-only and needs flashinfer's vendored CUTLASS closure | **custom CUTLASS (integrated, oracle-verified; FP32 SIMT, FP16/BF16 tensorop FP32-accumulate)** |
| silu_multiply | FP8-quantization fusions only | `act_and_mul_kernel` requires fused gate‖up `[tokens, 2d]` input; contract carries separate gate/up | **minimal owned kernel (last resort, oracle-verified)** |
| residual | fused into allreduce+rmsnorm (collective path) | fused into `FusedAddRMSNorm` only | **minimal owned kernel (last resort, oracle-verified)** |
| embedding | no embedding lookup at pin | no embedding lookup at pin | **minimal owned kernel (last resort, oracle-verified)** |
| attention | SM90/100/103 warp-specialized kernels behind its one-arch-per-module CMake (`HPC_KNOWN_ARCHS 90 100 103`); unavailable on SM89 by construction | decode/prefill device templates are paged-KV based (`paged_kv_t`, scheduler params); mapping the contiguous-BSHD contract is pending | **none yet — probes report Unimplemented with reason; M4 platform layer remains the execution path** |

Verification on the SM89 runner (RTX 4080 SUPER, CUDA 13.0): the flashinfer
RMSNorm and RoPE wrappers, the CUTLASS GEMM, and the three minimal owned
kernels all pass the CPU-oracle tests (`inferx_kernels_cuda_test`, 7/7), and
chain selection resolves rms_norm/rope → flashinfer, gemm → cutlass,
embedding/silu_multiply/residual → owned.

## Evidence and gaps

- hpc-ops builds one module per architecture and supports only SM90/100/103;
  its clean-header APIs (`rope.h`, `group_gemm.h`, `sampler.h`, `fuse_moe.h`,
  attention sm90) are the integration surface once SM90+ hardware qualifies.
  The provider probes stay registered at chain position 1 and flip when the
  adapter lands — no upper-layer change.
- flashinfer's norm/pos-enc device templates compile AOT under CUDA 13 for
  SM89 with the pinned CCCL on the include path (its default JIT path and
  cubin loader are not used and must stay out of the closure).
- CUTLASS 4.8 still ships the classic `gemm::device::Gemm`; tensorop
  instantiations require `cutlass::half_t`/`bfloat16_t` element types.
- Remaining qualification work: compute-sanitizer runs (all four tools),
  multi-stream races, tail shapes, FP16/BF16 oracle sweeps, and binary-size
  accounting for the integrated providers; the flashinfer attention paged-KV
  mapping; an SM90 machine for the hpc-ops adapters.

## Alternatives

- Keeping the ADR 0029 total rejection would leave every operator on owned
  kernels — exactly the maintenance cost this architecture exists to avoid.
- Vendoring patched copies of provider sources was rejected: dependency
  policy forbids local patches; integration happens at the recorded pins.

## Addendum 2026-09-02: TokenSpeed gap closure and head-to-head benchmark

The operator set was extended against the TokenSpeed reference inventory
(`tokenspeed-kernel/python/tokenspeed_kernel/ops`), following the same
hpc-ops -> flashinfer -> custom hierarchy:

| operator | provider integrated | status |
|---|---|---|
| silu_and_mul / gelu(_tanh)_and_mul (fused [T,2D]) | flashinfer `act_and_mul_kernel` | oracle-verified |
| fused_add_rmsnorm | flashinfer `FusedAddRMSNormKernel` | oracle-verified |
| gemma_rmsnorm | flashinfer `RMSNormKernel` (weight_bias=1) | oracle-verified |
| qk_rmsnorm | flashinfer `QKRMSNormKernel` | oracle-verified |
| attention decode/prefill | flashinfer `BatchDecodeWithPagedKVCacheDispatched` / paged prefill (contiguous-BSHD mapped onto `paged_kv_t`, workspace-marshalled) | GPU-verified (16-bit only) |
| top_p_renorm | flashinfer `TopPRenormProbKernel` (FP32 only at the pin) | oracle-verified |
| argmax, top_k_renorm, fp8 quant (tensor/token/group), moe softmax-topk, sigmoid-bias-topk, hadamard-128, add3, attn_res, hyperconnection mix/combine, mhc pre/post, ring sconv | owned (no external provider at the pins) | oracle-verified |

Benchmark harnesses (same CUDA-event methodology and workload matrix):
`benchmarks/kernels/kernels_benchmark.cc` and
`tools/bench/run_tokenspeed_bench.py`; results and plots under
`docs/benchmarks/`. Representative SM89 medians (TokenSpeed/InferX ratio,
>1 means InferX faster): shared flashinfer kernels sit at parity (1.01-1.11
— validates the AOT-instantiation approach); attention decode via the
flashinfer kernel beats TokenSpeed's Triton fallback 1.2-1.9x; qk_rmsnorm
3.8x (Triton fallback vs flashinfer kernel); moe softmax-topk 1.4x and
hadamard 21x vs Triton fallbacks. Regressions recorded honestly: dense gemm
vs cuBLASLt 0.78x (large) / 0.17x (skinny M=8; no GEMV path yet), owned
argmax 0.26x vs torch's vectorized reduction, owned fp8 quant 0.70x vs
Triton, rope 0.93x vs Triton's fused cache path.

Environment notes: the flashinfer wheel's sampling JIT cannot compile under
CUDA 13 (bundled CCCL removed `BlockAdjacentDifference::FlagHeads`), so
TokenSpeed's top_p_renorm comparator could not run; deep_gemm and
fast_hadamard_transform wheels do not build on this platform and are
import-stubbed (never on measured paths). Full NCU metric exports live in
`docs/benchmarks/ncu/`.
