# InferX kernels layer

Unified, backend-abstracted kernel execution (ADR 0031). Upper layers launch
operators through `include/inferx/kernels/ops/<category>.h`; dispatch selects
the registered backend by `DeviceKind` and then resolves a provider through
the fixed preference chain **hpc-ops → flashinfer → custom CUTLASS →
vendor/owned**, gated per request and compute capability (ADR 0032 records
what is integrated at the pinned revisions).

```
kernels/
├── CMakeLists.txt            # neutral target + CUDA-gated categories
├── include/inferx/kernels/   # unified API — backend-free (policy-tested)
│   ├── kernel_dispatch.h     # KernelBackend, KernelExecutionContext, registry
│   ├── provider.h            # ProviderId + preference chain
│   └── ops/<category>.h      # per-category launch entry points
├── src/                      # neutral dispatcher (builds on CPU presets)
└── cuda/
    ├── dispatch/             # CUDA backend, providers, shared op validation
    ├── include/              # internal launch shims (Abseil-free)
    └── ops/<category>/       # kernels by category (see below)
```

## Categories

`activation` `attention` `attn_res` `communication` `conv` `embedding`
`gemm` `hyperconnection` `kvcache` `layernorm` `metadata` `mhc` `moe`
`quantization` `sampling` `transform`

Current state (SM89, ADR 0032):

- **layernorm** — FlashInfer `RMSNormKernel`, ahead-of-time instantiation.
- **transform** — FlashInfer `BatchQKApplyRotaryPosIdsKernel` (NeoX
  non-interleave; head dims 32/64/128/256).
- **gemm** — custom CUTLASS classic GEMM (FP32 SIMT, FP16/BF16 tensorop with
  FP32 accumulation); neither provider library serves dense FP32/FP16 at the
  pins (hpc-ops is FP8/SM90+, flashinfer's runner is BF16-only).
- **activation**, **embedding** — minimal last-resort owned kernels; no
  external provider matches the contracts (fused layouts / FP8-only /
  absent at both pins).
- **attention** — FlashInfer paged decode/prefill mapped from the
  contiguous BSHD cache (16-bit dtypes; FP32 unsupported by the pinned
  kernels).
- **sampling**/**quantization**/**moe** — flashinfer top_p_renorm plus owned
  argmax, top_k_renorm, fp8 quantization, and routing kernels.
- **attn_res**, **hyperconnection**, **mhc**, **conv** — owned
  implementations of the TokenSpeed fused-residual and convolution ops.
- **communication**, **kvcache** (transfers), **metadata** — README
  scaffolds: multi-GPU or CUDA-graph infrastructure, out of scope on a
  single-GPU runner.

Head-to-head benchmark vs TokenSpeed on the same workloads:
`docs/benchmarks/RESULTS.md`.

## Adding a backend (AMD/NPU later)

Implement `inferx::kernels::KernelBackend` for the new `DeviceKind`, add
`kernels/<backend>/ops/<category>/` following the CUDA layout, and register
the backend at startup. Nothing above `kernels/` changes.

## Build

- CPU presets build the neutral dispatcher only (zero CUDA discovery).
- `cuda-release` adds the CUDA backend with owned kernels.
- `cuda-kernels` additionally enables the flashinfer/cutlass/hpc-ops
  provider options (requires `python3 tools/deps/bootstrap.py --profile core`
  plus `--dependency cutlass flashinfer hpc-ops`).
