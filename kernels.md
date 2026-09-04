# InferX Kernel Implementation Status

This document is the source of truth for the **InferX-owned** kernel layer
needed to execute the models registered by TokenSpeed. TokenSpeed supplies the
audited execution requirements and compatibility target; it does not own the
public include path, C++ namespace, build targets, or source tree. The analysis
is based on TokenSpeed commit
[`5af23865`](https://github.com/lightseekorg/tokenspeed/tree/5af23865b91afaf22d20f5d1172f3f1c45dc09ef)
(2026-09-03), not on a generic LLM checklist. Candidate CUDA providers were
checked at the revisions already pinned by InferX: HPC-Ops
[`f39028d9`](https://github.com/Tencent/hpc-ops/tree/f39028d9f5ab77f71906fbf929d1b611859ab6b7)
and FlashInfer
[`44428003`](https://github.com/flashinfer-ai/flashinfer/tree/44428003ba219c14b5473fefec8f7bfd4b72178e).

**Total required logical kernels/operators: 108**

Priority:

- P0: 18
- P1: 25
- P2: 63
- P3: 2

CUDA primary implementation ownership (one owner per logical operation):

- Reuse from HPC-Ops directly: 0
- Reuse from HPC-Ops with adapter: 6
- Reuse from FlashInfer: 25
- Reuse from the standard CUDA stack or another already-present provider: 15
- Custom CUTLASS/CuTe/CUDA implementation required: 62
- Unsupported/TBD: 0

ROCm:

- Existing reusable implementation: 95
- Custom implementation required: 10
- Unsupported/TBD: 3

NPU:

- Existing reusable implementation: 10
- InferX-owned implementation committed: 1 (K66, pending CANN qualification)
- Unsupported/TBD: 97

These are coverage counts, not a count of source files or launches.  For example, residual-add RMSNorm is an optimized K13 variant, gate/up and down expert GEMMs are K71 variants, and a fused sampling pipeline implements several of K81--K87.  Provider counts are deliberately disjoint: a fallback or optional second provider does not increment coverage.

### Ownership and naming

- Public headers live under `kernels/include/inferx/kernels/`.
- Public C++ symbols live in `inferx::kernels`; exported CMake targets use the
  `inferx::` alias namespace and build options use the existing `INFERX_*`
  convention.
- `kernels/common/` owns validation and backend dispatch. Backend code stays in
  `kernels/cuda/`, `kernels/rocm/`, or `kernels/ascend/` (the NPU backend is
  realized against the Ascend CANN runtime; see the NPU plan below).
- TokenSpeed names and paths appear only as execution evidence, compatibility
  requirements, or optional external-provider names. They are not part of the
  InferX kernel ABI or source ownership hierarchy.

## Audit scope and execution trace

The model registry dynamically imports every module exporting `EntryClass` in `python/tokenspeed/runtime/models/registry.py`.  The audited set includes DeepSeek V3/V3.2/V4 and NextN/DSpark variants, DFlash/DFlash2/DSpark, GLM5 and GLM5.3 Flash, GPT-OSS, Inkling, Kimi K2.5/K3, Llama/Eagle3, LongCat Flash, MiniMax M3, Qwen2, Qwen3 dense/MoE/ASR/Omni, Qwen3.5 dense/MoE, and Qwen4 experimental causal/conditional/NextN classes.  Vision/audio component paths reachable from conditional models were included.

The end-to-end trace used for the inventory is:

```text
engine/scheduler
  -> runtime/execution/model_runner.py:ModelRunner.forward
  -> runtime/models/*:EntryClass.forward
  -> runtime/models/base/{causal_lm,transformer_model,decoder_layer}.py
  -> runtime/layers/{linear,layernorm,activation,moe,attention,...}
  -> tokenspeed-kernel/python/tokenspeed_kernel/ops/* registry dispatch
  -> logits_processor.py
  -> runtime/sampling/backends/*
  -> runtime/distributed/* and KV-cache lifecycle
```

Static inspection covered all model modules, layer/runtime imports from `tokenspeed_kernel`, explicit `torch`/`torch.nn.functional` tensor operations, attention backends and cache recipes, sampling, speculative draft verification,
distributed communication, and the CUDA/ROCm/Ascend provider registrations. A static scan finds 72 literal `@register_kernel(family, mode)` pairs; programmatic registrations add provider variants. Neither registry number is
the inventory because registrations include planning/reference variants and omit unregistered PyTorch, cache-management, communication, convolution, PLE, and layout work.

Primary evidence anchors are
`python/tokenspeed/runtime/models/registry.py`,
`python/tokenspeed/runtime/execution/model_runner.py`,
`python/tokenspeed/runtime/layers/attention/registry.py`,
`python/tokenspeed/runtime/layers/attention/backends/`,
`python/tokenspeed/runtime/layers/attention/kv_cache/`,
`python/tokenspeed/runtime/sampling/backends/`,
`python/tokenspeed/runtime/distributed/`, and the public operation
families under `tokenspeed-kernel/python/tokenspeed_kernel/ops/`. Backend evidence comes from `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/`, `tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/`, HPC-Ops `hpc/` and `src/`, and FlashInfer `flashinfer/`; provider versions are the pinned commits at the top of this file.

### Counting boundary

- Count a separate logical operation when TokenSpeed has materially different inputs/state/semantics (for example dense MHA, paged extend, paged decode, MLA, DSA, KDA, MSA, and QSA).
- Do not count provider variants, tuning plans, graph capture, allocation, or warm-up as kernels.
- Do not count a fusion twice.  The unfused semantics remain the logical operations; fusion opportunities are recorded below.
- Include PyTorch operations when they lie on a registered model/runtime path; `F.embedding`, `torch.index_select`, `Conv2d`, and collectives are not exempt.
- P0 is the smallest single-GPU BF16 `LlamaForCausalLM` greedy path, including chunked prefill and paged decode.  Other registered models and quantization modes are P2; production-serving work is P1.

Three rows are explicitly **performance-operation contracts**, rather than an inflated count of their internal launches: K43 is TokenSpeed's atomic DSV4 index/cache-format producer, K54 is its candidate-state-producing KDA decode/verify contract, and K96 carries collective stream/aliasing semantics. Their correctness fallbacks compose K15/K20/K22/K61, K53/K99, and
K91/K13/K19 respectively. All other add+norm, GEMM+activation, MoE, and sampling fusions are implementation variants under existing K IDs and add nothing to the total.

Category reconciliation (sums to 108):

| Category | Count | Category | Count | Category | Count |
|---|---:|---|---:|---|---:|
| Activation | 4 | Attention | 20 | Cache | 7 |
| Communication | 8 | Draft model | 1 | Layout | 2 |
| Linear | 8 | Linear attention | 8 | Logits | 1 |
| Memory | 1 | Metadata | 1 | MoE | 8 |
| Multimodal | 2 | Norm | 4 | PLE | 3 |
| Position | 2 | Quantization | 5 | Reduction | 1 |
| Residual | 3 | Sampling | 10 | Sparse index | 7 |
| State/conv | 1 | Transform | 1 | **Total** | **108** |

## Coverage matrix

Abbreviations: CUDA `H` = HPC-Ops direct, `HA` = HPC-Ops adapter, `F` = FlashInfer, `S` = standard CUDA/already-present library, `C` = custom. ROCm `R` = reusable HIP/rocBLAS/hipBLASLt/Composable Kernel/Triton-Gluon/RCCL path, `C` = custom, `T` = unresolved. NPU `E` = implementation proven in the checked-in Ascend package, `I` = InferX-owned Ascend implementation committed but not yet CANN-qualified, and `T` = pending scope/format decisions. `TODO` means the backend-neutral wrapper does not yet exist in this repository, `NEEDS VERIFICATION` means it is implemented and exercised by out-of-tree drivers but awaits in-repo contract tests, and `PARTIAL` means the P0 subset of the operation is implemented while named variants remain.

Path prefixes in this table are real repository paths: `layers/` means `python/tokenspeed/runtime/layers/`, `models/` means
`python/tokenspeed/runtime/models/`, `attention/` means `python/tokenspeed/runtime/layers/attention/`, `sampling/`, `execution/` and `distributed/` are below `python/tokenspeed/runtime/`, and `tk/ops/` means `tokenspeed-kernel/python/tokenspeed_kernel/ops/`.

| ID | Logical kernel | Category | TokenSpeed call site | CUDA | ROCm | NPU | Preferred CUDA source | Pri | Status |
|---|---|---|---|---:|---:|---:|---|---:|---|
| K01 | Token embedding | Linear | `layers/vocab_parallel_embedding.py:VocabParallelEmbedding.forward` | C | R | T | custom gather kernel (`cuda/embedding/embedding.cu`) | P0 | NEEDS VERIFICATION |
| K02 | Vocab-parallel LM head/logits | Linear | `layers/vocab_parallel_embedding.py:ParallelLMHead.forward`; `layers/logits_processor.py` | S | R | T | cuBLASLt + gather adapter | P0 | TODO |
| K03 | Dense GEMM/linear | Linear | `layers/linear.py`; `layers/dense/unquant.py` -> `tk/ops/gemm.mm` | S | R | T | cuBLAS `cublasGemmEx` (dynamic-loaded) | P0 | NEEDS VERIFICATION |
| K04 | FP8/W8A8 dense GEMM | Linear | `layers/dense/{fp8,w8a8_fp8}.py` | F | R | T | FlashInfer FP8 GEMM | P1 | TODO |
| K05 | Four-bit dense GEMM | Linear | `layers/dense/{nvfp4,mxfp4}.py`; `quantization/compressed_tensors` | F | C | T | FlashInfer FP4; Marlin fallback | P2 | TODO |
| K06 | Batched GEMM | Linear | `tk/ops/gemm.bmm`; MLA/attention projections | F | R | T | FlashInfer BMM | P1 | TODO |
| K07 | Decode/routed GEMV | Linear | `layers/dense/unquant.py`; `tk/ops/gemm/{triton_gemv,routed_gemv}.py` | C | R | T | custom CUDA/CuTe | P1 | TODO |
| K08 | High-accuracy BF16 x FP32 GEMM | Linear | DSA/DSV4 indexer projections via `tk/ops/gemm` | HA | C | T | HPC-Ops `gemm_bf16xfp32` adapter | P2 | TODO |
| K09 | Tensor reshape/pack/split/concat/transpose | Layout | `models/llama.py:LlamaAttention.forward`; `models/qwen3_vision.py` | C | C | T | K66 block-copy fast path + custom strided-copy kernel | P0 | NEEDS VERIFICATION |
| K10 | Indexed gather/scatter | Layout | `layers/moe/expert.py`; `layers/logits_processor.py`; `attention/kv_cache/*` | C | R | T | custom indexed-copy kernel (`cuda/scatter/gather_scatter.cu`) | P0 | NEEDS VERIFICATION |
| K11 | Reductions/scans/prefix sums | Reduction | `layers/moe/topk.py`; `sampling/backends/triton_full.py`; `attention/backends/cache_metadata.py` | C | R | T | single-block chained-tile scan (`cuda/scan/scan.cu`) | P0 | NEEDS VERIFICATION |
| K12 | Page-table/cache-location construction | Cache | `execution/block_table.py`; `execution/cache_loc_kernel.py`; `attention/page_table.py` | C | R | T | locate kernel over K11 offsets (`cuda/metadata/metadata.cu`) | P0 | NEEDS VERIFICATION |
| K13 | RMSNorm (residual-add fused variant) | Norm | `layers/layernorm.py:RMSNorm.forward` -> `tk/ops/layernorm.rmsnorm` | C | R | E | block-per-row kernel (`cuda/norm/rmsnorm.cu`); FlashInfer optional | P0 | NEEDS VERIFICATION |
| K14 | LayerNorm/Gemma norm | Norm | `layers/layernorm.py`; `attention/linear/layernorm_gated.py` | F | R | T | FlashInfer norm | P2 | TODO |
| K15 | Per-head QK RMSNorm | Norm | `models/{qwen3,minimax_m3,inkling}.py`; `tk/ops/layernorm.qk_rmsnorm` | C | R | E | custom CUDA/CuTe | P2 | TODO |
| K16 | Grouped Gemma RMSNorm | Norm | `models/gpt_oss.py`; `tk/ops/layernorm` grouped norm | C | R | T | CUDA elementwise/reduction | P2 | TODO |
| K17 | SwiGLU (`silu_and_mul`) | Activation | `layers/activation.py:SiluAndMul.forward` | C | R | T | custom elementwise kernel (`cuda/activation/activation.cu`); FlashInfer optional | P0 | NEEDS VERIFICATION |
| K18 | SiTU and clamped/OpenAI SwiGLU | Activation | GPT-OSS, Kimi/DeepSeek model MLPs via `layers/activation.py` | C | R | T | CUDA/CuTe elementwise | P2 | TODO |
| K19 | Sigmoid/tanh gates and residual elementwise | Activation | `models/base/decoder_layer.py`; `layers/activation.py`; `attention/backends/{hybrid_linear_attn,hybrid_kda}.py` | C | R | T | residual-add kernel (`cuda/activation/activation.cu`); gates pending | P0 | PARTIAL |
| K20 | Standard RoPE/mRoPE | Position | `layers/rotary_embedding.py` -> `tk/ops/embedding` | C | R | E | pair-rotation kernel (`cuda/rope/rope.cu`); FlashInfer optional; mRoPE pending | P0 | PARTIAL |
| K21 | MLA RoPE + FP8 query/KV packing | Position | `tk/ops/embedding.mla_rope_quantize_fp8` | F | R | T | FlashInfer adapter | P2 | TODO |
| K22 | Hadamard transform | Transform | `models/{deepseek_v4,glm53_flash}.py`; `tk/ops/transform` | C | R | T | custom CUDA/CuTe | P2 | TODO |
| K23 | GELU/QuickGELU | Activation | `models/{moonvit,qwen3_vision,qwen3_audio}.py` | S | R | T | cuDNN/ATen | P2 | TODO |
| K24 | Dense MHA prefill/context | Attention | `attention/backends/mha.py`; `mm_encoder_attention.py` -> `mha_prefill` | HA | R | E | HPC-Ops + contract adapter | P0 | TODO |
| K25 | Paged MHA extend | Attention | `attention/backends/mha.py:forward_extend` | F | R | E | FlashInfer TRT-LLM adapter | P0 | TODO |
| K26 | Paged MHA decode | Attention | `attention/backends/mha.py:forward_decode` | F | R | E | FlashInfer TRT-LLM adapter | P0 | TODO |
| K27 | Relative-bias MHA prefill | Attention | Qwen4 path -> `relative_mha_prefill` | C | R | T | custom CuTe | P2 | TODO |
| K28 | Relative-bias MHA extend | Attention | Qwen4 path -> `relative_mha_extend_with_kvcache` | C | R | T | custom CuTe | P2 | TODO |
| K29 | Relative-bias MHA decode | Attention | Qwen4 path -> `relative_mha_decode_with_kvcache` | C | R | T | custom CuTe | P2 | TODO |
| K30 | MLA query normalize/project | Attention | `attention/backends/mla.py` -> `mla_normalize_project_query` | C | R | T | custom CUTLASS/CuTe | P2 | TODO |
| K31 | MLA value/output project | Attention | `attention/backends/mla.py` -> `mla_project_value` | C | R | T | custom CUTLASS/CuTe | P2 | TODO |
| K32 | MLA prefill | Attention | `attention/backends/{mla,flashmla,tokenspeed_mla}.py` | F | R | T | FlashInfer MLA | P2 | TODO |
| K33 | MLA paged extend | Attention | `attention/backends/{mla,flashmla,tokenspeed_mla}.py` -> `mla_extend_with_kvcache` | F | R | T | FlashInfer MLA | P2 | TODO |
| K34 | MLA paged decode | Attention | `attention/backends/{mla,flashmla,tokenspeed_mla}.py` -> `mla_decode_with_kvcache` | F | R | T | FlashInfer MLA | P2 | TODO |
| K35 | Attention partial-state merge | Attention | MLA/DSA split KV -> `attention.merge_state` | F | R | T | FlashInfer cascade merge | P1 | TODO |
| K36 | MSA paged extend | Attention | `attention/backends/msa.py` | C | C | T | custom CuTe/CUDA | P2 | TODO |
| K37 | MSA paged decode | Attention | `attention/backends/msa.py` | C | C | T | custom CuTe/CUDA | P2 | TODO |
| K38 | KPool index-cache update | Sparse index | `attention/kpool.py`; `attention/backends/{dsa,deepseek_v4}.py` | C | R | T | CUDA/CuTe | P2 | TODO |
| K39 | KPool top-k candidate select/map | Sparse index | `attention/kpool.py` -> kpool top-k ops | C | R | T | CUDA/CUB | P2 | TODO |
| K40 | DSA sparse prefill | Attention | `attention/backends/dsa.py` -> `dsa_prefill` | F | R | T | FlashInfer TRT-LLM DSA | P2 | TODO |
| K41 | DSA sparse decode | Attention | `attention/backends/dsa.py` -> `dsa_decode` | F | R | T | FlashInfer TRT-LLM DSA | P2 | TODO |
| K42 | DSA index top-k | Sparse index | `attention/backends/dsa.py` -> `dsa_{prefill,decode}_topk` | C | R | T | CUDA/CUB | P2 | TODO |
| K43 | DSV4 SWA QK norm+RoPE+FP8 cache insert | Attention | `attention/backends/deepseek_v4.py`; `deepseek_v4_ops.py` | C | R | T | custom CuTe/CUDA | P2 | TODO |
| K44 | DSV4 compressed index cache | Sparse index | `attention/deepseek_v4_ops.py` | C | R | T | custom CUTLASS/CUDA | P2 | TODO |
| K45 | DSV4 sparse top-k/metadata | Sparse index | `attention/deepseek_v4_ops.py`; `deepseek_v4/metadata.py` | C | R | T | custom CUDA/CUB | P2 | TODO |
| K46 | DSV4 selected prefill attention | Attention | `attention/backends/deepseek_v4.py` | C | R | T | custom CuTe | P2 | TODO |
| K47 | DSV4 selected decode attention | Attention | `attention/backends/deepseek_v4.py` | C | R | T | custom CuTe | P2 | TODO |
| K48 | GDN chunk prefill | Linear attention | `attention/linear/gdn.py` -> `gdn_chunk_prefill` | F | R | T | FlashInfer GDN | P2 | TODO |
| K49 | GDN decode | Linear attention | `attention/linear/gdn.py` -> `gdn_decode_step` | F | R | T | FlashInfer GDN | P2 | TODO |
| K50 | GDN multi-token verify | Linear attention | speculative path -> `gdn_decode_mtp` | F | R | T | FlashInfer GDN | P2 | TODO |
| K51 | GDN accepted-state replay/commit | Linear attention | `hybrid_linear_attn.py` -> `gdn_replay_commit` | C | R | T | CUDA/CuTe | P3 | TODO |
| K52 | KDA paged prefill | Linear attention | `attention/backends/hybrid_kda.py` -> `kda_paged_prefill` | C | R | T | custom CuTe/CUDA | P2 | TODO |
| K53 | KDA paged decode | Linear attention | `attention/backends/hybrid_kda.py` -> `kda_paged_decode` | C | R | T | custom CuTe/CUDA | P2 | TODO |
| K54 | KDA fused decode/verify conv+state | Linear attention | `attention/backends/hybrid_kda.py` -> fused KDA dispatch | C | R | T | custom CuTe/CUDA | P2 | TODO |
| K55 | KDA accepted-state replay/commit | Linear attention | `attention/backends/hybrid_kda.py` -> `kda_replay_commit` | C | R | T | CUDA/CuTe | P3 | TODO |
| K56 | QSA index/cache pipeline | Sparse index | `attention/qsa/indexer.py`; `backends/qsa.py` | C | T | T | CUDA/CuTe | P2 | TODO |
| K57 | QSA block top-k | Sparse index | `attention/qsa/indexer.py` -> QSA block top-k ops | C | T | T | CUDA/CUB | P2 | TODO |
| K58 | QSA sparse attention | Attention | `attention/backends/qsa.py` | C | T | T | custom CuTe | P2 | TODO |
| K59 | Paged MHA KV-cache write | Cache | `attention/kv_cache/mha.py:set_kv_buffer` | C | R | T | CUDA/CuTe | P0 | TODO |
| K60 | MLA KV-cache read/write | Cache | `attention/kv_cache/mla.py` | C | R | T | CUDA/CuTe | P2 | TODO |
| K61 | Quantized KV write + scale layout | Cache | `attention/kv_cache/{mha,dsa,mla}.py`; `models/dflash.py` | C | R | T | CUDA/CuTe | P1 | TODO |
| K62 | State-page verify/copy/commit | Cache | hybrid cache classes; `draft_page_staging.py` | C | R | T | CUDA | P1 | TODO |
| K63 | Host/device cache-range transfer | Cache | `attention/kv_cache/recipes/transfer.py`; `cache/transfer` | S | R | T | CUDA memcpy/streams | P1 | TODO |
| K64 | Cache-group table unpack/decode locations | Cache | `attention/backends/cache_groups.py`; `attention/kv_cache/recipes/*.py` | C | R | T | CUDA/CUB | P1 | TODO |
| K65 | Metadata tape/sequence-index/tau generation | Metadata | `attention/backends/cache_metadata.py`; `ops/metadata`; Inkling `log_scaling_tau` | C | R | T | seq-index + tau kernels (`cuda/metadata/metadata.cu`); tape pending | P0 | PARTIAL |
| K66 | Generic device fill/copy/zero | Memory | `attention/kv_cache/arena.py`; `execution/{input_buffer,workspace}.py` | S | R | I | `kernels/cuda/memory/device_memory.cu` | P0 | NEEDS VERIFICATION |
| K67 | Softmax top-k MoE routing | MoE | `layers/moe/topk.py` -> `moe_softmax_topk` | C | R | T | CUDA/CUB | P2 | TODO |
| K68 | Specialized sigmoid/bias/group/sink MoE routing | MoE | `layers/moe/topk.py` -> sigmoid/Inkling/MiniMax top-k APIs | C | R | T | CUDA/CUB | P2 | TODO |
| K69 | DSV4 sqrt-softplus/hash routing | MoE | `models/deepseek_v4.py`; `tk/ops/moe` | C | R | T | CUDA/CUB | P2 | TODO |
| K70 | Expert token permute/count/sort | MoE | `layers/moe/expert.py`; `tk/ops/moe.moe_plan` | C | R | T | CUDA/CUB | P2 | TODO |
| K71 | MoE grouped GEMM | MoE | `layers/moe/expert.py` -> `moe_apply`; gate/up and down | HA | R | T | HPC-Ops grouped GEMM adapter | P2 | TODO |
| K72 | Expert activation/requantization | MoE | `tk/ops/moe` between expert GEMMs | HA | R | T | HPC-Ops activation adapter | P2 | TODO |
| K73 | Token unpermute/weighted reduce/finalize shared | MoE | `layers/moe/expert.py`; `tk/ops/moe` finalize | HA | R | T | HPC-Ops fused MoE adapter | P2 | TODO |
| K74 | Latent-MoE projections/fused tail | MoE | `layers/moe/latent.py`; `models/{kimi_k3,kimi_k3_comm,deepseek_v4}.py` | C | R | T | custom CUTLASS/CuTe | P2 | TODO |
| K75 | FP8 quantize/dequantize/scales | Quantization | `layers/quantization/fp8.py`; `tk/ops/quantization` | C | R | T | custom CUDA/CuTe | P1 | TODO |
| K76 | MXFP8 quantize/dequantize | Quantization | `tk/ops/quantization/__init__.py:quantize_mxfp8`; MHA/MoE callers | F | R | T | FlashInfer (SM100+) | P2 | TODO |
| K77 | NVFP4 quantize/dequantize | Quantization | `layers/quantization/nvfp4.py` | F | C | T | FlashInfer (Blackwell) | P2 | TODO |
| K78 | MXFP4 quantize/dequantize | Quantization | `layers/quantization/mxfp4.py` | F | R | T | FlashInfer (Blackwell) | P2 | TODO |
| K79 | GPTQ/Marlin weight repack | Quantization | `compressed_tensors/gptq_marlin_moe.py` | S | C | T | existing Marlin CUDA | P2 | TODO |
| K80 | Logit softcap + grammar mask | Logits | `layers/logits_processor.py`; grammar backend mask | C | R | T | CUDA elementwise | P1 | TODO |
| K81 | Logits penalties/bias | Sampling | `sampling/backends/{triton,triton_full}.py` | C | R | T | CUDA/CUB fused pipeline | P1 | TODO |
| K82 | Token-count update | Sampling | `sampling/backends/{triton,triton_full}.py` count/update kernels | C | R | T | CUDA/CUB | P1 | TODO |
| K83 | Softmax/log-softmax | Sampling | `layers/logits_processor.py`; `sampling/backends/{flashinfer,triton_full}.py` | F | R | T | FlashInfer softmax | P1 | TODO |
| K84 | Top-k | Sampling | `sampling/backends/{flashinfer,triton,triton_full}.py`; `layers/logits_processor.py` | F | R | T | FlashInfer sampling | P1 | TODO |
| K85 | Top-p | Sampling | `sampling/backends/{flashinfer,triton,triton_full}.py` | F | R | T | FlashInfer sampling | P1 | TODO |
| K86 | Min-p | Sampling | `sampling/backends/{flashinfer,triton,triton_full}.py` | F | R | T | FlashInfer sampling | P1 | TODO |
| K87 | Stochastic/Gumbel sampling | Sampling | `sampling/backends/{flashinfer,flashinfer_full,triton,triton_full}.py` | F | R | T | FlashInfer sampling + adapter | P1 | TODO |
| K88 | Greedy argmax (local) | Sampling | `sampling/backends/greedy.py` | S | R | T | CUB/ATen | P0 | TODO |
| K89 | Selected/top logprobs | Sampling | `layers/logits_processor.py`; `engine/logprobs.py` | C | R | T | CUDA/CUB | P1 | TODO |
| K90 | Speculative chain verify/sample | Sampling | `sampling/backends/{flashinfer,triton}.py`; `execution/drafter/*` | F | C | T | FlashInfer speculative sampling | P2 | TODO |
| K91 | All-reduce | Communication | `distributed/comm_ops.py` | S | R | E | NCCL | P1 | TODO |
| K92 | All-gather | Communication | `distributed/comm_ops.py`; `layers/logits_processor.py`; `layers/linear.py` | S | R | E | NCCL | P1 | TODO |
| K93 | Reduce-scatter | Communication | `distributed/comm_ops.py` | S | R | E | NCCL | P1 | TODO |
| K94 | All-to-all expert dispatch | Communication | `distributed/comm_ops.py:all_to_all_single`; `layers/moe/expert.py` | S | R | T | NCCL/DeepEP optional | P2 | TODO |
| K95 | Send/recv/broadcast | Communication | `distributed/comm_ops.py:{pp_send,pp_recv}`; `device_communicators/pynccl.py` | S | R | E | NCCL | P1 | TODO |
| K96 | Fused collective+residual+norm | Communication | `distributed/comm_backend/*allreduce*`; fused norm path | HA | C | T | HPC-Ops adapter | P1 | TODO |
| K97 | Distributed argmax | Communication | `layers/logits_processor.py`; `tk/ops/sampling/cute_dsl.py` | C | C | T | custom NCCL+CUDA | P1 | TODO |
| K98 | DP sampling gather/swap | Communication | `distributed/dp_sampling_{comm,swap}.py` | C | R | T | CUDA+NCCL | P1 | TODO |
| K99 | Causal/depthwise Conv1D + recurrent state | State/conv | `attention/linear/causal_conv1d.py`; KDA/GDN models | C | R | T | CUDA/CuTe | P2 | TODO |
| K100 | PLE n-gram hash | PLE | `layers/qwen4_exp_ple.py` -> PLE kernel API | C | R | T | CUDA | P2 | TODO |
| K101 | PLE page gather/scatter | PLE | `layers/qwen4_exp_ple.py` -> paged PLE gather/scatter APIs | C | R | T | CUDA/CUB | P2 | TODO |
| K102 | PLE dilated convolution | PLE | `layers/qwen4_exp_ple.py` -> PLE convolution APIs | C | R | T | CUDA/CuTe | P2 | TODO |
| K103 | mHC pre/post Sinkhorn mapping | Residual | `layers/hyperconnection.py`; GLM/Kimi models | C | R | T | CUDA/CuTe | P2 | TODO |
| K104 | Gated residual hyperconnection | Residual | `layers/hyperconnection.py` | C | R | T | CUDA/CuTe | P2 | TODO |
| K105 | AttnRes | Residual | `models/{minimax_m3,kimi_k3}.py`; `tk/ops/attn_res` | C | R | T | CUDA/CuTe | P2 | TODO |
| K106 | Multimodal Conv2D/Conv3D patch projection | Multimodal | `models/{moonvit,qwen3_vision,qwen3_audio,glm53_flash}.py` | S | R | T | cuDNN | P2 | TODO |
| K107 | Multimodal pooling/interpolation/patch merge | Multimodal | `models/{moonvit,qwen3_vision,qwen3_audio,glm53_flash}.py`; `execution/multimodal_runtime.py` | S | R | T | cuDNN/ATen | P2 | TODO |
| K108 | DFlash2 grouped dynamic convolution/transition | Draft model | `models/dflash2.py`; `execution/drafter/dflash2.py` | S | R | T | ATen/cuDNN baseline | P2 | TODO |

The CUDA ownership totals above are derived directly from this matrix: `H=0`, `HA=6`, `F=25`, `S=15`, and `C=62`.  ROCm similarly has `R=95`, `C=10`, and `T=3`; NPU has `E=10`, `I=1`, and `T=97`.

## HPC-Ops compatibility audit

The following table is exhaustive: every K ID appears in exactly one decision row.  “Direct” still requires the thin C++ backend wrapper; “adapter” means a semantic/layout adapter rather than modification of TokenSpeed model code.

| Decision | K IDs | HPC-Ops implementation and source | Dtype/layout | Applicable path | Limitations / adapter work |
|---|---|---|---|---|---|
| **USE** | none | No HPC-Ops API accepts a complete TokenSpeed operation contract without adaptation. | n/a | n/a | A thin backend wrapper alone is not counted as an adapter; layout/metadata/weight/topology conversion is. |
| **USE WITH ADAPTER** | K08 | `hpc/gemm.py:gemm_bf16xfp32`; `src/gemm/sm90/gemm.cu` | BF16 activation; caller supplies FP32 weight decomposed into high/low BF16 terms plus FP32 scale; row-major 2-D | DSA/DSV4 accuracy-sensitive FP32-output projection | Prepare/cache `w_high` and `w_low` at load time, allocate/zero split-K flags, and validate exact FP32 output tolerance; built architectures are SM90/100/103. |
| **USE WITH ADAPTER** | K24 | `hpc/attention.py:attention_prefill_bf16` | packed `[T,H,D]`, BF16; causal sequence metadata | Llama/Qwen dense prefill when `D=128` | Convert TokenSpeed cumulative-sequence metadata; dispatch elsewhere for FP16, other head dimensions, sinks/window/logit-cap/LSE. |
| **USE WITH ADAPTER** | K71 | `hpc/group_gemm.py` FP8 per-tensor/blockwise group GEMM | grouped row-major FP8 with explicit scales/counts | FP8 expert gate/up and down GEMMs | Translate `moe_plan` sorting/count metadata and scale layout; BF16/FP4 experts need other paths. |
| **USE WITH ADAPTER** | K72--K73 | `hpc/act.py`; `hpc/fuse_moe.py` gather/reduce/finalize | BF16/FP8, HPC routing-index layout | low-latency FP8 MoE | Adapter must preserve TokenSpeed top-k weights, expert ownership and shared-expert semantics; shape coverage must be benchmarked. |
| **USE WITH ADAPTER** | K96 | `hpc/allreduce.py` | BF16, single-node NVLink, hidden 4096/5120/7168 paths | TP all-reduce + residual + RMSNorm | Own multicast/P2P workspaces and barriers; only use on qualified topology/shapes, fall back to NCCL + K13. |
| **NOT SUITABLE** | K13, K17, K20, K25--K26, K38--K42, K67--K68, K75, K81--K87, K91--K95, K97--K98, K103--K105 | similarly named code exists in `normalization.py`, `act.py`, `rope.py`, `attention.py`, `topk.py`, `sampler.py`, `allreduce.py`, `iHC.py`, or `stem.py` | mostly BF16/FP8 and specialized layouts | none as primary owner | Plain RMSNorm is absent (only norm+FP8 quant); RoPE assumes packed QKV and fused KV store; paged attention is head-dim/page/GQA constrained and lacks full sink/logit-cap/LSE contract; sampler lacks min-p, frequency/presence bias, grammar and logprobs; allreduce is not general NCCL; `topk` does not compute DSA scores; `stem` is not a TokenSpeed path; iHC semantics are not TokenSpeed Sinkhorn/hyperconnection semantics. |
| **NOT AVAILABLE** | K01--K07, K09--K12, K14--K16, K18--K19, K21--K23, K27--K37, K43--K66, K69--K70, K74, K76--K80, K88--K90, K99--K102, K106--K108, excluding IDs above | no matching HPC-Ops API at the pinned revision | n/a | n/a | A similarly named building block, if any, does not expose the required TokenSpeed operation contract. |

Important exception to the requested priority rule: FlashInfer owns K25/K26. HPC-Ops paged attention is promising as an optional fast path, but the pinned source constrains important variants (head dimension 128; BF16 pages commonly
32/64, FP8 page 64; FP8 GQA ratio 4/8; limited MTP) and does not satisfy the complete TokenSpeed window/sink/logit-cap/LSE contract.  FlashInfer already has a TokenSpeed TRT-LLM adapter with wider dispatch coverage, so choosing HPC-Ops
as the only implementation would regress supported models.

## FlashInfer compatibility audit

This audit uses both the pinned FlashInfer source and TokenSpeed's own adapter code under `tk/ops/attention/flashinfer`, `tk/ops/gemm/flashinfer`, `tk/ops/moe/flashinfer`, and sampling/norm/quantization registrations.  That is
stronger evidence than matching names in documentation.

| Decision | K IDs | FlashInfer source/API and layout | Dtype/architecture | Applicable path and limitations |
|---|---|---|---|---|
| **USE** | K04--K06 | `flashinfer/gemm/gemm_base.py:{mm_fp8,mm_mxfp8,bmm_fp8,bmm_mxfp8}` plus `flashinfer/gemm/gemm_bf16_fp4.py`; bound by `tk/ops/gemm/flashinfer.py` | BF16/FP16, FP8; FP4 is Blackwell-specific | Dense quantized GEMM and BMM; retain cuBLASLt/CUTLASS fallback outside supported shapes and scale layouts. |
| **USE** | K13--K14, K17, K20 | `flashinfer/norm/__init__.py:{rmsnorm,fused_add_rmsnorm}`, `flashinfer/activation.py:{silu_and_mul,gelu_and_mul}`, `flashinfer/rope.py:apply_rope*`; bound by TokenSpeed's layernorm/activation/embedding adapters | FP16/BF16; architecture varies by kernel | Exact common inference semantics are present; Gemma variants require epsilon/layout tests. |
| **USE WITH ADAPTER** | K21 | `flashinfer/rope.py:mla_rope_quantize_fp8`, bound by `tk/ops/embedding/flashinfer.py` | BF16 input, FP8 packed output; modern NVIDIA GPUs | Bind positions/scales and the compressed-KV layout; validate scale granularity. |
| **USE WITH ADAPTER (FALLBACK)** | K24 | `flashinfer/prefill.py:{BatchPrefillWithRaggedKVCacheWrapper,BatchPrefillWithPagedKVCacheWrapper}` | packed/ragged Q/K/V, FP16/BF16; head/architecture dispatch varies | Use when HPC-Ops rejects the tuple; supports GQA, window, soft-cap, LSE and sink variants through capability-specific plans. Convert cumulative lengths once and test every option combination. K24 remains counted under HPC-Ops primary ownership. |
| **USE WITH ADAPTER** | K25--K26 | `flashinfer/prefill.py:BatchPrefillWithPagedKVCacheWrapper`, `flashinfer/decode.py:BatchDecodeWithPagedKVCacheWrapper`, and TRT-LLM entry points bound in `tk/ops/attention/flashinfer/__init__.py`; cache becomes `[page,H,page_size,D]` after adapter permutation | FP16/BF16/FP8; optimized Hopper/SM100+; extend head dims 64/128/256 | Supports causal/window/sinks in covered dispatches. Logit cap and LSE remain fallbacks; exact page/head/GQA tuple is a runtime capability key. |
| **USE WITH ADAPTER** | K32--K34 | `flashinfer/mla/_batch_mla/_wrapper.py:BatchMLAPagedAttentionWrapper` and `flashinfer/mla/_core.py:trtllm_batch_decode_with_kv_cache_mla`, imported by `tk/ops/attention/flashinfer/__init__.py` | BF16/FP16 and compressed latent layouts checked by the runners | Map TokenSpeed page table/sequence metadata and projection mode; do not mix MHA and MLA cache ABI. |
| **USE** | K35 | `flashinfer/cascade.py:{merge_state,merge_state_in_place,merge_states}` | FP16/BF16 state plus FP32 LSE | Exact partial `(value,lse)` merge semantics; validate strides. |
| **USE WITH ADAPTER** | K40--K41 | FlashInfer TRT-LLM sparse-MLA APIs in `flashinfer/mla/_core.py`, registered by `tk/ops/attention/flashinfer/__init__.py` | SM100+, page 64; BF16/FP8 Q; qk dims 128/192 or NoPE 256; latent 512; selected top-k 512/1024/2048/2051 | Proven DSA path; no logit cap/LSE. Dispatch must reject unsupported tuples. |
| **USE** | K48--K50 | `flashinfer/gdn_prefill.py:chunk_gated_delta_rule` and `flashinfer/gdn_decode.py:{gated_delta_rule_decode,gated_delta_rule_mtp}`, bound by `tk/ops/attention/flashinfer/gated_delta_rule.py` | FP16/BF16; state/conv layouts fixed by adapter | GDN chunk, decode, and MTP verify. Replay commit K51 is not provided by this API. |
| **USE WITH ADAPTER (FALLBACK)** | K71--K73 | FlashInfer grouped/fused MoE APIs under `flashinfer/fused_moe/` and `flashinfer/gemm/gemm_base.py:group_gemm_*`, bound by `tk/ops/moe/flashinfer/` | BF16/FP8/FP4 coverage depends on runner and SM; provider routing/count layout | Use for formats/shapes outside HPC-Ops, especially Blackwell FP4. Translate K70 plan and preserve K17/K18 activation, scale and shared-expert finalize semantics. These IDs remain counted under HPC-Ops primary ownership. |
| **USE** | K76--K78 | `flashinfer/quantization/fp8_quantization.py:{mxfp8_quantize,mxfp8_grouped_quantize}` and `flashinfer/quantization/fp4_quantization.py:{nvfp4_quantize,mxfp4_quantize}`, bound by `tk/ops/quantization/flashinfer.py` | MXFP8 and FP4 paths require SM100/Blackwell as registered | Use capability gates; never silently reinterpret NVFP4 as MXFP4. |
| **USE** | K83--K87 | `flashinfer/sampling.py:{sampling_from_logits,top_k_top_p_sampling_from_logits,top_p_renorm_probs,top_k_renorm_probs,min_p_sampling_from_probs}` through `tk/ops/sampling/flashinfer.py` | FP16/BF16/FP32 logits depending API | TokenSpeed adapters prove basic semantics; K81 penalties and K82 counts remain separate. Gumbel behavior needs deterministic-seed tests. |
| **USE** | K90 | `flashinfer/sampling.py:chain_speculative_sampling` | CUDA; check installed FlashInfer version | Fits TokenSpeed chain verification; state-page commit remains K62. |
| **NOT SUITABLE** | K03, K07--K08, K15--K16, K18--K19, K22--K23, K27--K31, K36--K39, K42--K47, K51--K70, K74--K75, K79--K82, K88--K89, K91--K108 | available building blocks do not expose the exact operation, or HPC-Ops/standard CUDA has priority | varies | No primary selection. In particular FlashInfer has no routed small-M BF16 GEMV matching TokenSpeed's measured route; its public quantization covers MXFP8/FP4 but not TokenSpeed's complete ordinary FP8 conversion/scale contract; its generic paged append consumes batch/position/page metadata rather than TokenSpeed's direct flat-slot write ABI; and generic sparse attention does not establish QSA/DSV4 semantics. |
| **NOT AVAILABLE** | K01--K02, K09--K12 | no useful standalone FlashInfer ownership | n/a | Use standard CUDA or custom metadata kernels. |

All `USE` rows still carry **NEEDS VERIFICATION** until a shape-contract test has covered dtype, strides, head dimensions, GQA/MQA ratio, page size, quantization scales, architecture, batching mode, prefill/decode semantics, and CUDA graph capture.  A provider's runtime capability predicate must reject, not coerce, an unsupported tuple.

## Backend-specific plan

### CUDA custom work

The 62 `C` rows are the custom CUDA backlog:

```text
K01, K07, K09-K13, K15, K16, K18, K19, K22,
K27-K31, K36-K39, K42-K47, K51-K62,
K64-K65, K67-K70, K74-K75, K80-K82, K89, K97-K105
```

- Use CUTLASS/CuTe for projection/GEMM-coupled work (K30, K31, K44, K74), custom attention/state-space kernels (K27--K29, K36--K37, K43, K46--K47, K51--K58), and specialized convolution (K99, K102).
- Use CUDA C++ plus CUB for page metadata, cache scatter, routing, scans, top-k, logit processing, logprobs and distributed argmax (K12, K38--K39, K42--K45, K59--K62, K64--K65, K67--K70, K75, K80--K82, K89, K97--K101) and plain CUDA C++ for the strided layout copy (K09).
- Use small CUDA/CuTe elementwise-reduction kernels for norms, gates, Hadamard and residual maps (K15--K16, K18--K19, K22, K103--K105).
- Every custom choice exists because neither audited provider exposes the complete operation contract.  CUTLASS/CuTe/CUB are implementation libraries, not model-visible dependencies.

### ROCm

ROCm correctness can reuse 95 logical operations from PyTorch ROCm, rocBLAS, hipBLASLt, MIOpen, RCCL, portable TokenSpeed Triton, or the checked-in `tokenspeed-kernel-amd` package.  That package explicitly covers MHA, relative
MHA, MLA, DSA, KDA, GEMM, BF16/MXFP4 MoE, and sampling for gfx950 and a smaller gfx1250 subset.  `R` therefore means a source/API exists, not that every GPU and shape already has production tuning.

Custom ROCm work is K05 (the combined four-bit contract needs format-specific
dispatch), K08 (BF16 x FP32 accuracy contract), K09 (strided layout copy), K36--K37
(MSA), K77 (NVFP4), K79 (Marlin-format repack/GEMM), K90 (chain speculative
verify), K96 (fused collective+norm), and K97 (distributed argmax).  Implement
with hipBLASLt or
Composable Kernel for GEMM, HIP/Triton-Gluon for fusion/attention, and RCCL for
collectives.  K56--K58 QSA are `T`: portable source exists, but the audited AMD
package does not register or test the exact QSA contract.  Do not treat CUDA
source compatibility as evidence.

ROCm qualification must span gfx950 and gfx1250 separately and record wave
size, LDS use, FP8/FP4 encoding, matrix-core layout, page size, graph capture,
and RCCL topology.  A generic PyTorch ROCm fallback is acceptable for
correctness but does not close a P1 performance item.

### NPU

The repository defines this backend as Ascend, and the kernels layer now
commits to the Ascend CANN runtime: backend code lives in `kernels/ascend/`,
builds behind `INFERX_ENABLE_ASCEND` (CANN discovery via
`INFERX_ASCEND_CANN_PATH` or `ASCEND_TOOLKIT_HOME`), and links AscendCL.  The
`Backend` enumeration names the concrete runtime (`kAscend`), mirroring
`kCuda`/`kRocm`, because runtimes are source-incompatible; the operation
contracts above remain runtime-neutral.  The first InferX-owned NPU row is
K66: `ascend/memory/device_memory.cc` maps fill/zero/copy to
`aclrtMemsetAsync`/`aclrtMemcpyAsync`, uses the bytewise path for
repeated-byte typed patterns, and rejects non-repeated typed fills with
`kUnsupported` until an AscendC pattern kernel is integrated (reject, never
coerce).  It is committed but not CANN-qualified, so it counts as `I`, not
`E`.

The proven NPU implementations inherited from TokenSpeed evidence remain:

- K13 RMSNorm and its residual-add variant through
  `tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/layernorm.py` using `torch_npu.npu_rms_norm` and
  `npu_add_rms_norm`;
- K15 QK RMSNorm as composed NPU tensor operations;
- K20 RoPE through `tokenspeed_kernel_npu/ops/rotary_embedding.py` and
  `torch_npu.npu_mrope`;
- K24--K26 through `tokenspeed_kernel_npu/ops/mha.py` and
  `torch_npu.npu_fused_infer_attention_score`; and
- K91--K93 and K95 through `runtime/distributed/comm_backend/hccl.py`, which
  implements all-reduce, all-gather, reduce-scatter, send and receive through
  `torch.distributed` HCCL; broadcast call sites use the same process group.
  Its even-split `all_to_all_single` does not prove K94's variable expert
  dispatch contract.

The attention registrations accept FP16/BF16; paged extend/decode accept page
64 or 128; decode is single-query-token; and the checked contract explicitly
rejects sliding windows, sinks, logit cap, returned LSE, scaled FP8 cache, and
fused KV write.  Consequently the other 97 rows remain **TBD - depends on
Ascend scope decisions**: model subset, aclnn/AscendC kernel strategy, tensor
formats, collectives, and quantization formats.  The runtime question is now
settled; before assigning custom work, fix that scope.  The ten proven
operations stay plugin evidence (`torch_npu`/HCCL) outside the C++ core until
each is re-qualified behind this repository's gates.

## Detailed kernel specifications

The matrix supplies category, exact source call site, priority, and owner.  The
entries below add the implementation contract.  Shapes use `T` tokens, `B`
sequences, `Hq/Hkv` heads, `D` head width, `P` pages, `S` page size, `M/K/N`
GEMM dimensions, `E` experts and `V` vocabulary.  Dense floating paths must
support at least BF16 for P0; quantized types are called out explicitly.  Unless
an NPU entry is one of K13/K15/K20/K24--K26/K91--K93/K95, or K66 (the
InferX-owned Ascend implementation above), its NPU disposition is the global
TBD above.  `TODO` means implementation/wrapper and contract tests remain.

### K01: Token embedding

- **Category / priority:** Linear / P0.

- **Usage:** `VocabParallelEmbedding.forward`; token IDs enter every text model
  before `TransformerModel.forward`.
- **Semantics:** gather `[T]` integer IDs from `[local_V,Dmodel]`, mask IDs
  outside the TP shard, return `[T,Dmodel]`; contiguous FP16/BF16 weights.
- **Backends:** CUDA ATen/custom gather; ROCm PyTorch ROCm; NPU TBD. HPC-Ops and
- **CUDA (committed):** `embedding.h:EmbeddingGatherAsync` CuTe tensor gather kernel; out-of-shard IDs produce zero rows.
  FlashInfer: **NOT AVAILABLE**.
- **Performance:** every request, bandwidth-bound; fuse shard mask/reduction
  only behind the same ABI. **Status: NEEDS VERIFICATION.**

### K02: Vocab-parallel LM head/logits

- **Category / priority:** Linear / P0.

- **Usage:** `ParallelLMHead.forward` then `LogitsProcessor.forward` on selected
  hidden rows for all causal/conditional models.
- **Semantics:** `[Tsel,Dmodel] x [local_V,Dmodel]^T -> [Tsel,local_V]`, optional
  TP gather; BF16/FP16 input/weight, FP32 logits where requested.
- **Backends:** CUDA cuBLASLt/CUTLASS plus NCCL; ROCm hipBLASLt+RCCL; NPU TBD.
  HPC-Ops/FlashInfer: **NOT AVAILABLE** as the full sharded operation.
- **Performance:** once per sampled/score position; large-V bandwidth/compute
  risk. **Status: TODO.**

### K03: Dense GEMM/linear

- **Category / priority:** Linear / P0.

- **Usage:** `layers/linear.py` and `dense/unquant.py`; all QKV, O, MLP and
  projector layers call `tokenspeed_kernel.ops.gemm.mm` or linear.
- **Semantics:** row-major `[M,K] x [N,K]^T -> [M,N]`, optional bias/output
  dtype; arbitrary leading dimensions flattened, FP16/BF16 minimum.
- **Backends:** CUDA cuBLASLt first, CUTLASS tuned shapes; ROCm hipBLASLt/rocBLAS;
- **CUDA (committed):** `gemm.h:DenseGemmAsync` via `cublasGemmEx` (FP32 accumulate, row-major `[M,K]x[N,K]^T`); cuBLAS is dynamic-loaded after CUDA warm-up because linking it directly breaks device enumeration on WSL2/CUDA 12.0.
  NPU TBD. HPC-Ops lacks generic GEMM; FlashInfer is not primary.
- **Performance:** several times per layer/token; highest roofline exposure;
  fuse bias/activation only as a variant. **Status: NEEDS VERIFICATION.**

### K04: FP8/W8A8 dense GEMM

- **Category / priority:** Linear / P1.

- **Usage:** `dense/fp8.py`, `dense/w8a8_fp8.py`, and FP8 quantization methods.
- **Semantics:** FP8 activation/weight matrix product with per-tensor or
  blockwise scales and BF16/FP16 output; scale layout is part of the ABI.
- **Backends:** CUDA FlashInfer with CUTLASS fallback; ROCm hipBLASLt/CK/Triton;
  NPU TBD. HPC-Ops generic implementation: **NOT AVAILABLE**.
- **Performance:** every quantized linear, P1; fuse K75 input quantization when
  profitable. **Status: NEEDS VERIFICATION.**

### K05: Four-bit dense GEMM

- **Category / priority:** Linear / P2.

- **Usage:** `dense/nvfp4.py`, `dense/mxfp4.py`, and compressed-tensor W4A16
  schemes.
- **Semantics:** format-tagged NVFP4, MXFP4, or packed GPTQ weights with their
  exact block scales/zeros; never reinterpret formats; BF16/FP16 output.
- **Backends:** CUDA FlashInfer FP4 and existing Marlin for W4A16; ROCm custom
  format dispatch/CK; NPU TBD. HPC-Ops: **NOT AVAILABLE**.
- **Performance:** per quantized linear; Blackwell-only FlashInfer variants;
  P2 additional modes. **Status: NEEDS VERIFICATION.**

### K06: Batched GEMM

- **Category / priority:** Linear / P1.

- **Usage:** `gemm.bmm` in attention/MLA projections and grouped tensor paths.
- **Semantics:** strided or pointer-array `[B,M,K] x [B,K,N] -> [B,M,N]`, with
  explicit transpose/stride and BF16/FP16 accumulation rules.
- **Backends:** CUDA FlashInfer BMM (cuBLASLt fallback); ROCm hipBLASLt; NPU
  TBD. HPC-Ops: **NOT AVAILABLE**.
- **Performance:** repeated in specialized attention; preserve strides to avoid
  K09 materialization. **Status: NEEDS VERIFICATION.**

### K07: Decode/routed GEMV

- **Category / priority:** Linear / P1.

- **Usage:** `gemm.decode_gemv` and low-token routed expert/projection paths.
- **Semantics:** small-M GEMM/GEMV, sometimes indexed by expert, with the same
  numerical result and scale contract as K03/K04.
- **Backends:** custom CUDA/CuTe using TokenSpeed's measured row-CTA/skinny-GEMM
  routes as reference; ROCm Gluon/Triton/hipBLASLt; NPU TBD. HPC-Ops is **NOT
  AVAILABLE** and FlashInfer has no matching routed BF16 small-M contract.
- **Performance:** every decode layer; latency critical and graph-capturable.
  **Status: NEEDS VERIFICATION.**

### K08: High-accuracy BF16 x FP32 GEMM

- **Category / priority:** Linear / P2.

- **Usage:** DSA/DSV4 router and state-compress projections where TokenSpeed
  requests high-accuracy GEMM.
- **Semantics:** `[M,K]` BF16 times `[N,K]` FP32 with FP32-like error and
  selectable BF16/FP32 output; ordinary BF16 downcast is not equivalent.
- **Backends:** CUDA HPC-Ops `gemm_bf16xfp32` **USE WITH ADAPTER**; ROCm custom
  CK/HIP; NPU TBD. Adapter prepares/caches high/low BF16 weights and split-K
  flags, then validates the FP32 output tolerance.
- **Performance:** each sparse-index pass, accuracy/performance risk.
  **Status: ADAPTER REQUIRED.**

### K09: Tensor reshape/pack/split/concat/transpose

- **Category / priority:** Layout / P0.

- **Usage:** QKV construction, TP partitions, attention layouts, MoE and all
  conditional encoders throughout `models/*`.
- **Semantics:** view when strides permit; otherwise typed copies among packed
  token/head/channel layouts, preserving ordering and arbitrary valid strides.
- **Backends:** implemented behind `kernels/include/inferx/kernels/layout.h`:
  `TensorView` (non-owning pointer, `DType`, rank, shape, element strides)
  with host-side metadata transforms `TransposeView`, `SliceView` (covers
  split), and `ReshapeView` (C-contiguous sources only; other strides return
  `kUnsupported` and the caller copies first), plus `MakeContiguousView` and
  the device operations `CopyLayoutAsync` (strided elementwise copy that
  falls back to a K66 block copy when both sides are contiguous) and
  `ConcatAsync` (per-source copies into destination slices, conservative
  overlap rejection, rank capped at `kMaxTensorRank`). CUDA and ROCm run a
  rank-specialized CuTe strided-copy kernel (`cuda/layout/strided_copy.cu`,
  `rocm/layout/strided_copy.hip`); Ascend serves only the contiguous block
  path through AscendCL and rejects strided tuples with `kUnsupported` until
  an AscendC kernel is integrated. Neither specialist provider owns this
  family.
- **Performance:** many times per layer; eliminate copies or fuse into producer/
  consumer; the strided kernel is correctness-first (per-element index
  decomposition) and needs vectorized inner-dim specialization before
  performance claims. **Status: NEEDS VERIFICATION** — contract checks pass
  on all builds and CUDA on-device checks pass including graph capture and
  transpose/concat round trips; ROCm needs a HIP runner and the Ascend
  strided path is unimplemented.

### K10: Indexed gather/scatter

- **Category / priority:** Layout / P0.

- **Usage:** MoE token routing, selected logits, page/state updates and
  multimodal token merge.
- **Semantics:** gather/scatter rows by int32/int64 indices, duplicate-index
  behavior explicit; FP16/BF16/FP8 payload and arbitrary row width.
- **Backends:** CUDA ATen/CUB; ROCm equivalent; NPU TBD. HPC-Ops/FlashInfer do
- **CUDA (committed):** `scatter.h:GatherRowsAsync`/`ScatterRowsAsync` CuTe tensor kernels over any payload dtype with int32/int64 indices; scatter duplicate winner is explicitly unspecified.
  not cover the general contract.
- **Performance:** common bandwidth kernel; fuse with K70/K73 where safe.
  **Status: NEEDS VERIFICATION.**

### K11: Reductions/scans/prefix sums

- **Category / priority:** Reduction / P0.

- **Usage:** MoE counts/offsets, sampling statistics, metadata, masks and
  sequence/cache offsets.
- **Semantics:** sum/max and exclusive/inclusive scans over rows or token lists;
  stable FP32 accumulation where probabilities depend on it.
- **Backends:** CUDA CUB/ATen; ROCm rocPRIM/ATen; NPU TBD. Providers not needed.
- **CUDA (committed):** `scan.h:ExclusiveSumAsync`/`InclusiveSumAsync` as a CuTe tensor-backed single-block chained-tile scan — exact, deterministic, workspace-free; a multi-block variant is future work for huge counts.
- **Performance:** frequent small launches; expose typed primitives internally,
  not as model-level virtual calls. **Status: NEEDS VERIFICATION.**

### K12: Page-table/cache-location construction

- **Category / priority:** Cache / P0.

- **Usage:** `BlockTable`, `cache_loc_kernel`, `attention/page_table.py` before
  paged attention/cache writes.
- **Semantics:** map request/token offsets to int32 page IDs and in-page slots;
  honor speculative staging and invalid sentinels.
- **Backends:** custom CUDA+CUB; ROCm portable Triton/rocPRIM; NPU TBD. No
- **CUDA (committed):** `metadata.h:BuildCacheLocationsAsync` composes K11 offsets with a CuTe tensor-backed binary-search locate kernel and the `-1` out-of-range sentinel.
  provider API matches.
- **Performance:** every batch preparation; graph-friendly, no host round-trip.
  **Status: NEEDS VERIFICATION.**

### K13: RMSNorm

- **Category / priority:** Norm / P0.

- **Usage:** `RMSNorm.forward` in every decoder; optional residual input returns
  normalized output and updated residual.
- **Semantics:** rowwise `x*rsqrt(mean(x^2)+eps)*weight`, FP32 reduction over
  `[T,D]`; residual-add fusion is an implementation variant.
- **Backends:** CUDA FlashInfer; ROCm Triton/Gluon; Ascend `npu_rms_norm` /
- **CUDA (committed):** `norm.h:RmsnormAsync`/`AddRmsnormAsync` CuTe tensor block-per-row kernels with FP32 statistics and fused residual update; FlashInfer remains an optional accelerator.
  `npu_add_rms_norm`. HPC-Ops norm+FP8-only API is **NOT SUITABLE**.
- **Performance:** twice or more per layer; fuse residual, K75, or collective
  without changing semantics. **Status: NEEDS VERIFICATION.**

### K14: LayerNorm/Gemma norm

- **Category / priority:** Norm / P2.

- **Usage:** vision/audio blocks, gated linear attention, and Gemma/GPT-OSS
  variants in `layers/layernorm.py`.
- **Semantics:** rowwise mean/variance LayerNorm plus Gemma weight-offset
  convention where configured; FP16/BF16 I/O, FP32 statistics.
- **Backends:** CUDA FlashInfer; ROCm Triton/ATen; NPU TBD. HPC-Ops lacks it.
- **Performance:** per applicable layer; fuse gate only as tested variant.
  **Status: NEEDS VERIFICATION.**

### K15: Per-head QK RMSNorm

- **Category / priority:** Norm / P2.

- **Usage:** attention modules normalize Q/K by head before RoPE/attention.
- **Semantics:** independent RMSNorm over last `D` of `[T,H,D]`, distinct Q and
  K weights/eps; BF16/FP16 with FP32 reduction.
- **Backends:** custom CUDA/CuTe; ROCm Triton; Ascend composed tensor ops.
  HPC-Ops fused packed-QKV RoPE contract is **NOT SUITABLE**.
- **Performance:** each applicable layer; fuse K20/K43. CUDA requires custom
  work. **Status: CUSTOM REQUIRED.**

### K16: Grouped Gemma RMSNorm

- **Category / priority:** Norm / P2.

- **Usage:** GPT-OSS grouped expert/attention normalization path.
- **Semantics:** Gemma-offset RMSNorm over independently grouped rows with
  group metadata and FP32 statistics.
- **Backends:** custom CUDA; ROCm Gluon/Triton; NPU TBD. No HPC-Ops equivalent;
  generic FlashInfer norm lacks grouped metadata.
- **Performance:** per GPT-OSS layer; fuse adjacent routing/quantization.
  **Status: CUSTOM REQUIRED.**

### K17: SwiGLU

- **Category / priority:** Activation / P0.

- **Usage:** `SiluAndMul.forward` in Llama/Qwen and many dense/shared MLPs.
- **Semantics:** split `[...,2I]`, return `silu(gate)*up` as `[...,I]`, preserve
  BF16/FP16 dtype and strided leading rows.
- **Backends:** CUDA FlashInfer; ROCm Triton/Gluon; NPU TBD. HPC-Ops activation
- **CUDA (committed):** `activation.h:SiluAndMulAsync` CuTe tensor elementwise kernel with FP32 math.
  is quantization-coupled and not primary.
- **Performance:** every MLP; fuse K03 output or K75. **Status: NEEDS VERIFICATION.**

### K18: SiTU and clamped/OpenAI SwiGLU

- **Category / priority:** Activation / P2.

- **Usage:** `layers/activation.py` from GPT-OSS, Kimi and DeepSeek model MLP/
  expert variants.
- **Semantics:** exact configured sigmoid/tanh/clamp formulas, limits and split
  order; these are not interchangeable with K17.
- **Backends:** custom CUDA/CuTe; ROCm Gluon/Triton; NPU TBD. Specialist
  provider formulas do not prove these semantics.
- **Performance:** each affected MLP/expert; fuse with K71 when format permits.
  **Status: CUSTOM REQUIRED.**

### K19: Sigmoid/tanh gates and residual elementwise

- **Category / priority:** Activation / P0.

- **Usage:** decoder residual adds on the P0 path plus GDN/KDA and model-specific
  gated residual calculations.
- **Semantics:** broadcast-safe add/multiply and specified sigmoid/tanh gates on
  equal-shaped hidden/state tensors; no implicit dtype promotion.
- **Backends:** custom/fused CUDA baseline, portable ROCm; NPU TBD. Providers do
- **CUDA (committed):** `activation.h:ResidualAddAsync` CuTe tensor kernel (the P0 decoder residual); sigmoid/tanh gate variants remain.
  not own the generic contract.
- **Performance:** many times per layer; aggressively fuse into norms and state
  updates. **Status: PARTIAL (residual add committed; gates pending).**

### K20: Standard RoPE/mRoPE

- **Category / priority:** Position / P0.

- **Usage:** dense attention in Llama/Qwen and multimodal-position variants via
  `tokenspeed_kernel.ops.rope`.
- **Semantics:** rotate configured Q/K dimensions by positions and cos/sin,
  supporting interleaved/non-interleaved and mRoPE sections.
- **Backends:** CUDA FlashInfer; ROCm Triton; Ascend `npu_mrope`. HPC-Ops packed
- **CUDA (committed):** `rope.h:ApplyRopeAsync` CuTe layout pair-rotation kernel supporting half-rotary and interleaved conventions, in-place and out-of-place, with tail pass-through; mRoPE sectioning remains.
  QKV+KV-store API is **NOT SUITABLE** for the general operation.
- **Performance:** each attention layer; fuse K15 and K59 when contract matches.
  **Status: PARTIAL (standard RoPE committed; mRoPE pending).**

### K21: MLA RoPE + FP8 packing

- **Category / priority:** Position / P2.

- **Usage:** `mla_rope_quantize_fp8`/`mla_nope_quantize_fp8` while preparing MLA
  query and compressed KV.
- **Semantics:** apply rotary subspace, concatenate NoPE latent fields, quantize
  with explicit scales into the cache/query layout.
- **Backends:** CUDA FlashInfer adapter; ROCm Triton; NPU TBD. HPC-Ops has no MLA
  contract.
- **Performance:** every MLA layer/token; scale and byte-layout compatibility
  are mandatory. **Status: ADAPTER REQUIRED.**

### K22: Hadamard transform

- **Category / priority:** Transform / P2.

- **Usage:** NVFP4/MXFP4 activation preprocessing and quantized expert paths.
- **Semantics:** normalized block Hadamard over the configured last-dimension
  tiles, including padding/permutation contract before quantization.
- **Backends:** custom CUDA/CuTe (existing TokenSpeed CUDA is reference);
  portable ROCm; NPU TBD. No matching provider API.
- **Performance:** every FP4 layer; fuse K77/K78. **Status: CUSTOM REQUIRED.**

### K23: GELU/QuickGELU

- **Category / priority:** Activation / P2.

- **Usage:** vision/audio encoder and projector MLPs in conditional models.
- **Semantics:** exact approximate/exact GELU or QuickGELU formula selected by
  model config, elementwise over FP16/BF16 tensors.
- **Backends:** CUDA cuDNN/ATen; ROCm MIOpen/ATen; NPU TBD. FlashInfer gated GELU
  is optional only when split layout matches.
- **Performance:** multimodal-only; fuse producer GEMM opportunistically.
  **Status: TODO.**

### K24: Dense MHA prefill/context

- **Category / priority:** Attention / P0.

- **Usage:** `MhaBackend.forward_prefill` and multimodal encoder attention call
  `mha_prefill` for packed variable-length Q/K/V.
- **Semantics:** packed `[T,Hq,D]`, `[T,Hkv,D]` with cumulative lengths, causal
  or noncausal, GQA/MQA, scale and optional window/sinks/logit-cap/LSE.
- **Backends:** CUDA HPC-Ops **USE WITH ADAPTER** for its BF16/D=128 qualified
  subset, FlashInfer ragged prefill adapter otherwise; custom fallback remains
  for any rejected tuple. ROCm Gluon/Triton; Ascend fused-infer-attention subset.
- **Performance:** every prefill layer; highest throughput risk. **Status:
  ADAPTER REQUIRED.**

### K25: Paged MHA extend

- **Category / priority:** Attention / P0.

- **Usage:** `MhaBackend.forward_extend` after new K/V are associated with the
  page table.
- **Semantics:** packed variable query plus paged cache `[P,S,Hkv,D]` (logical
  ABI), int32 page table and lengths; causal/window/GQA/sinks options.
- **Backends:** CUDA FlashInfer TRT-LLM adapter; ROCm Gluon/Triton; Ascend for
  FP16/BF16, pages 64/128 and restricted options. HPC-Ops is **NOT SUITABLE** as
  the complete implementation.
- **Performance:** every chunked/continuous prefill layer; cache-layout adapter
  must be zero-copy. **Status: ADAPTER REQUIRED.**

### K26: Paged MHA decode

- **Category / priority:** Attention / P0.

- **Usage:** `MhaBackend.forward_decode` for one or speculative multiple query
  positions per sequence.
- **Semantics:** Q `[B,Q,Hq,D]` or packed equivalent over paged K/V, GQA/MQA,
  sequence lengths, causal/window/sinks and optional LSE.
- **Backends:** CUDA FlashInfer TRT-LLM; ROCm Gluon/Triton; Ascend only `Q=1`,
  page 64/128, FP16/BF16 restricted options. HPC-Ops optional qualified fast
  path, not primary.
- **Performance:** every generated token/layer; highest latency and mixed-length
  scheduling risk. **Status: ADAPTER REQUIRED.**

### K27: Relative-bias MHA prefill

- **Category / priority:** Attention / P2.

- **Usage:** Qwen4 experimental attention calls `relative_mha_prefill`.
- **Semantics:** K24 plus per-query/key relative logits and optional temperature
  `tau`; bias broadcasting and causal masking are model-defined.
- **Backends:** custom CUDA CuTe; ROCm AMD Gluon `rmha`; NPU TBD. Generic dense
  attention in HPC-Ops/FlashInfer lacks this contract.
- **Performance:** each Qwen4 attention layer; fuse bias into online softmax.
  **Status: CUSTOM REQUIRED.**

### K28: Relative-bias MHA extend

- **Category / priority:** Attention / P2.

- **Usage:** Qwen4 extend path calls `relative_mha_extend_with_kvcache`.
- **Semantics:** K25 paged cache plus relative logits/tau aligned to each
  query-cache position; variable-length packed batches.
- **Backends:** custom CUDA CuTe; ROCm Gluon `rmha`; NPU TBD. No audited CUDA
  provider matches.
- **Performance:** continuous batching; avoid materializing full bias matrices.
  **Status: CUSTOM REQUIRED.**

### K29: Relative-bias MHA decode

- **Category / priority:** Attention / P2.

- **Usage:** Qwen4 decode path calls `relative_mha_decode_with_kvcache`.
- **Semantics:** K26 paged decode plus relative bias/tau for Q=1/MTP and GQA;
  stable online softmax.
- **Backends:** custom CUDA CuTe; ROCm AMD Gluon; NPU TBD. Providers unavailable.
- **Performance:** per Qwen4 token/layer, latency critical. **Status: CUSTOM
  REQUIRED.**

### K30: MLA query normalize/project

- **Category / priority:** Attention / P2.

- **Usage:** `MlaBackend` calls `mla_normalize_project_query` before MLA
  attention.
- **Semantics:** normalize compressed query, project into NoPE/RoPE and/or
  absorbed-head fields using model weights; output strides are attention ABI.
- **Backends:** custom CUTLASS/CuTe; ROCm hipBLASLt/Gluon; NPU TBD. Neither CUDA
  provider exposes the fused TokenSpeed projection.
- **Performance:** every MLA layer; fuse norm/projection/RoPE packing.
  **Status: CUSTOM REQUIRED.**

### K31: MLA value/output projection

- **Category / priority:** Attention / P2.

- **Usage:** `MlaBackend` calls `mla_project_value` after latent attention.
- **Semantics:** transform latent value heads to model head/output layout,
  accepting contiguous or provider-preferred weight orientation.
- **Backends:** custom CUTLASS/CuTe; ROCm hipBLASLt/Gluon; NPU TBD. Generic GEMM
  alone does not own the layout contract.
- **Performance:** every MLA layer; fold transpose into epilogue. **Status:
  CUSTOM REQUIRED.**

### K32: MLA prefill

- **Category / priority:** Attention / P2.

- **Usage:** MLA, FlashMLA and TokenSpeed-MLA backends dispatch `mla_prefill`.
- **Semantics:** causal packed attention over compressed latent KV and separate
  rotary fields, returning latent value state and optional LSE.
- **Backends:** CUDA FlashInfer MLA; ROCm AMD/Triton MLA; NPU TBD. HPC-Ops has no
  MLA implementation.
- **Performance:** DeepSeek/Kimi prefill hot path; validate latent/head dims and
  absorption mode. **Status: NEEDS VERIFICATION.**

### K33: MLA paged extend

- **Category / priority:** Attention / P2.

- **Usage:** MLA backends call `mla_extend_with_kvcache` for new prompt chunks.
- **Semantics:** packed query over paged compressed latent+RoPE cache with page
  table, lengths, causal mask and scale.
- **Backends:** CUDA FlashInfer adapter; ROCm AMD/Triton; NPU TBD.
- **Performance:** continuous batching on MLA models; no cache repack in hot
  path. **Status: ADAPTER REQUIRED.**

### K34: MLA paged decode

- **Category / priority:** Attention / P2.

- **Usage:** MLA backends call `mla_decode_with_kvcache` during generation.
- **Semantics:** Q=1/MTP query against paged compressed latent cache, producing
  absorbed latent output with stable softmax.
- **Backends:** CUDA FlashInfer adapter; ROCm AMD/Triton; NPU TBD.
- **Performance:** each MLA token/layer, high latency risk; page/head tuples must
  be capability-gated. **Status: ADAPTER REQUIRED.**

### K35: Attention partial-state merge

- **Category / priority:** Attention / P1.

- **Usage:** split-KV MLA/DSA and cascade paths call `attn_merge_state`.
- **Semantics:** merge partial `(O,LSE)` pairs using log-sum-exp identities;
  value FP16/BF16, statistics FP32, arbitrary split count.
- **Backends:** CUDA FlashInfer `merge_state`; ROCm Triton; NPU TBD. HPC-Ops
  internal combine is not a public general contract.
- **Performance:** split attention batches; fuse into provider when possible.
  **Status: NEEDS VERIFICATION.**

### K36: MSA paged extend

- **Category / priority:** Attention / P2.

- **Usage:** `MsaBackend.forward_extend` in MiniMax M3.
- **Semantics:** model-specific MSA query/state against a paged cache, with
  TokenSpeed's page size 128 and extend metadata.
- **Backends:** custom CUDA CuTe from the checked-in MSA source; custom ROCm;
  NPU TBD. Generic MHA/MLA is not semantically equivalent.
- **Performance:** every MiniMax extend layer; architecture-specialized and high
  porting risk. **Status: CUSTOM REQUIRED.**

### K37: MSA paged decode

- **Category / priority:** Attention / P2.

- **Usage:** `MsaBackend.forward_decode` in MiniMax M3.
- **Semantics:** single/MTP query MSA over page-128 state/cache with its custom
  head/state mapping.
- **Backends:** custom CUDA CuTe; custom ROCm; NPU TBD. No audited provider.
- **Performance:** every MiniMax token/layer; high latency/portability risk.
  **Status: CUSTOM REQUIRED.**

### K38: KPool index-cache update

- **Category / priority:** Sparse index / P2.

- **Usage:** `attention/kpool.py` appends/updates/tails keys for DSA and DSV4
  sparse indexing.
- **Semantics:** scatter compressed index keys to flat/paged slots, maintain
  valid tails and lengths, and return mapped locations.
- **Backends:** custom CUDA; ROCm Triton; NPU TBD. HPC-Ops top-k/stem does not
  implement cache mutation.
- **Performance:** every sparse-attention token; fuse quantize/write.
  **Status: CUSTOM REQUIRED.**

### K39: KPool top-k candidate select/map

- **Category / priority:** Sparse index / P2.

- **Usage:** `KPool` selects candidates and maps logical page offsets to flat KV
  slots before sparse attention.
- **Semantics:** score/select exact top-k over valid cached keys, deterministic
  tie policy, output int32 slots in attention-expected ordering.
- **Backends:** custom CUDA+CUB; ROCm Triton/rocPRIM; NPU TBD. HPC-Ops `topk` only
  solves row top-k and is **NOT SUITABLE** for this pipeline.
- **Performance:** each sparse token/layer; avoid global score materialization.
  **Status: CUSTOM REQUIRED.**

### K40: DSA sparse prefill

- **Category / priority:** Attention / P2.

- **Usage:** `DsaBackend` calls `dsa_prefill` with selected KV slots.
- **Semantics:** causal sparse attention over per-query selected cache indices,
  compressed latent/value fields and exact DSA scaling.
- **Backends:** CUDA FlashInfer TRT-LLM adapter for qualified SM100 tuples;
  ROCm AMD Gluon/Triton; NPU TBD. HPC sparse prefill uses a different block-mask
  contract.
- **Performance:** dominant long-context DSA prefill; custom fallback required
  for rejected tuples. **Status: NEEDS VERIFICATION.**

### K41: DSA sparse decode

- **Category / priority:** Attention / P2.

- **Usage:** `DsaBackend` calls `dsa_decode` per generated/MTP token.
- **Semantics:** attend to selected flat slots from page-64 DSA cache with
  DSA latent/head mapping and stable softmax.
- **Backends:** CUDA FlashInfer qualified SM100 adapter; ROCm AMD/Triton; NPU
  TBD. HPC decode lacks selected-index semantics.
- **Performance:** highest DSA decode risk; top-k 512/1024/2048/2051 and head
  tuples must be gated. **Status: NEEDS VERIFICATION.**

### K42: DSA index top-k

- **Category / priority:** Sparse index / P2.

- **Usage:** `dsa_prefill_topk`/`dsa_decode_topk` score query against index cache.
- **Semantics:** compute DSA-specific dot products (including FP8 cache scales),
  mask invalid/past entries, select/map exact top-k.
- **Backends:** custom CUDA/CUB; ROCm AMD Gluon/Triton; NPU TBD. Plain provider
  top-k is not sufficient.
- **Performance:** every DSA layer/token, often memory bound; fuse scoring and
  selection. **Status: CUSTOM REQUIRED.**

### K43: DSV4 SWA QK norm+RoPE+FP8 cache insert

- **Category / priority:** Attention / P2.

- **Usage:** `DeepseekV4Backend`/`deepseek_v4_ops.py` prepares sliding-window
  Q/K and writes the page-256 cache.
- **Semantics:** per-head norm, RoPE, scaling/FP8 quantization and paged insert
  with DSV4 head dimensions and window metadata.
- **Backends:** custom CUDA CuTe; ROCm AMD implementation; NPU TBD. HPC fused
  RoPE/store has incompatible packed/layout assumptions.
- **Performance:** every DSV4 layer/token; fusion is essential. **Status: CUSTOM
  REQUIRED.**

### K44: DSV4 compressed index cache

- **Category / priority:** Sparse index / P2.

- **Usage:** `deepseek_v4_ops.py` projects/quantizes state into the DSV4 index
  cache and updates it.
- **Semantics:** accuracy-sensitive projection followed by exact scale/layout
  packing at page slots; output consumed by K45.
- **Backends:** custom CUTLASS/CUDA (K08 may implement projection); ROCm AMD;
  NPU TBD. No complete provider operation.
- **Performance:** per DSV4 token/layer; fuse K08/K75/store. **Status: CUSTOM
  REQUIRED.**

### K45: DSV4 sparse top-k/metadata

- **Category / priority:** Sparse index / P2.

- **Usage:** `deepseek_v4_ops.py` and `deepseek_v4/metadata.py` select blocks/
  slots for K46/K47.
- **Semantics:** score valid index pages, apply causal/window rules, exact top-k,
  and emit provider-ready block metadata.
- **Backends:** custom CUDA+CUB; ROCm AMD; NPU TBD. Generic top-k/sparse
  attention is insufficient.
- **Performance:** every DSV4 attention call; high launch and bandwidth risk.
  **Status: CUSTOM REQUIRED.**

### K46: DSV4 selected prefill attention

- **Category / priority:** Attention / P2.

- **Usage:** `DeepseekV4Backend` prefill/extend path consumes K45 selections.
- **Semantics:** mixed sliding-window and selected long-context attention over
  DSV4 cache fields, page 256, with exact merge/scaling rules.
- **Backends:** custom CUDA CuTe; ROCm AMD; NPU TBD. No audited CUDA provider
  matches this hybrid contract.
- **Performance:** long-context DSV4 prefill; highest custom-attention risk.
  **Status: CUSTOM REQUIRED.**

### K47: DSV4 selected decode attention

- **Category / priority:** Attention / P2.

- **Usage:** `DeepseekV4Backend` decode consumes K45 metadata.
- **Semantics:** low-token selected+window attention over DSV4 paged cache,
  including MTP where selected by runtime.
- **Backends:** custom CUDA CuTe; ROCm AMD; NPU TBD.
- **Performance:** every DSV4 token/layer; highest latency risk. **Status: CUSTOM
  REQUIRED.**

### K48: GDN chunk prefill

- **Category / priority:** Linear attention / P2.

- **Usage:** `attention/linear/gdn.py` for Qwen3.5/GLM hybrid linear-attention
  layers.
- **Semantics:** chunkwise gated-delta recurrence over Q/K/V, gates, convolution
  state and matrix state; return outputs and updated state.
- **Backends:** CUDA FlashInfer gated-delta-rule; ROCm Triton; NPU TBD. HPC-Ops
  unavailable.
- **Performance:** each GDN prefill layer; state layout and FP32 accumulation
  must match. **Status: NEEDS VERIFICATION.**

### K49: GDN decode

- **Category / priority:** Linear attention / P2.

- **Usage:** `gdn_decode_step` for one generation token.
- **Semantics:** one recurrent delta update per batch/head using persisted state
  and gates, returning output and new state.
- **Backends:** CUDA FlashInfer; ROCm Triton; NPU TBD.
- **Performance:** every GDN decode layer, latency critical and in-place state
  ordering sensitive. **Status: NEEDS VERIFICATION.**

### K50: GDN multi-token verify

- **Category / priority:** Linear attention / P2.

- **Usage:** `gdn_decode_mtp` during speculative verification.
- **Semantics:** sequentially evaluate several proposed tokens without exposing
  unaccepted final state; output per-token values and candidate state.
- **Backends:** CUDA FlashInfer; ROCm Triton; NPU TBD.
- **Performance:** each GDN speculative step; semantic test must compare serial
  K49 execution. **Status: NEEDS VERIFICATION.**

### K51: GDN accepted-state replay/commit

- **Category / priority:** Linear attention / P3.

- **Usage:** `HybridLinearAttentionBackend` calls `gdn_replay_commit` after the
  accepted speculative prefix is known.
- **Semantics:** replay/commit exactly the accepted token count into persistent
  conv/matrix state; rejected tokens must leave no effect.
- **Backends:** custom CUDA; ROCm Triton; NPU TBD. FlashInfer's forward API does
  not supply the commit operation.
- **Performance:** speculative-only optimization; correctness/state isolation
  dominates. **Status: CUSTOM REQUIRED.**

### K52: KDA paged prefill

- **Category / priority:** Linear attention / P2.

- **Usage:** `HybridKdaBackend` calls `kda_paged_prefill` for Kimi K3 hybrid
  layers.
- **Semantics:** KDA chunk/recurrent computation over paged convolution and
  matrix state with per-sequence lengths.
- **Backends:** custom CUDA CuTe/checked-in KDA sources; ROCm AMD Gluon; NPU TBD.
  No matching HPC/FlashInfer public contract.
- **Performance:** KDA prefill hot path; recurrence and cache ABI are major
  risks. **Status: CUSTOM REQUIRED.**

### K53: KDA paged decode

- **Category / priority:** Linear attention / P2.

- **Usage:** `HybridKdaBackend` calls `kda_paged_decode`.
- **Semantics:** one-token recurrent KDA update/read over paged state, returning
  output and committed state.
- **Backends:** custom CUDA CuTe; ROCm AMD Gluon; NPU TBD.
- **Performance:** each Kimi K3 KDA layer/token; persistent state ordering and
  launch latency critical. **Status: CUSTOM REQUIRED.**

### K54: KDA fused decode/verify conv+state

- **Category / priority:** Linear attention / P2.

- **Usage:** hybrid KDA speculative dispatch selects fused conv/gate/recurrent
  decode or verify paths.
- **Semantics:** combine causal convolution window, gate precompute and KDA
  recurrence for one or multiple proposals, producing candidate state.
- **Backends:** custom CUDA CuTe/CUDA; ROCm AMD/Triton; NPU TBD.
- **Performance:** speculative KDA hot path; compare numerically to unfused K99
  plus K53. **Status: CUSTOM REQUIRED.**

### K55: KDA accepted-state replay/commit

- **Category / priority:** Linear attention / P3.

- **Usage:** `HybridKdaBackend` invokes replay after speculative acceptance.
- **Semantics:** commit only accepted convolution/recurrent state rows and
  preserve page ownership for rejected proposals.
- **Backends:** custom CUDA; ROCm Triton; NPU TBD.
- **Performance:** speculative-only; fuse with K62 if safe. **Status: CUSTOM
  REQUIRED.**

### K56: QSA index/cache pipeline

- **Category / priority:** Sparse index / P2.

- **Usage:** `attention/qsa/indexer.py` creates and updates QSA compressed index
  cache for the QSA backend.
- **Semantics:** model-specific query/key projection, normalization/
  quantization and page writes with QSA scale/layout metadata.
- **Backends:** custom CUDA/CuTe; ROCm **TBD** pending exact AMD registration and
  tests; NPU TBD. Generic sparse providers do not establish compatibility.
- **Performance:** every QSA layer/token; projection/write fusion. **Status:
  CUSTOM REQUIRED.**

### K57: QSA block top-k

- **Category / priority:** Sparse index / P2.

- **Usage:** QSA indexer selects candidate blocks/slots before attention.
- **Semantics:** score compressed QSA index by block, mask invalid/causal ranges,
  select deterministic top-k and emit the backend metadata layout.
- **Backends:** custom CUDA+CUB; ROCm/NPU TBD. HPC row top-k and FlashInfer
  generic sampling top-k are not suitable.
- **Performance:** each QSA call; fusion and selection scaling are risks.
  **Status: CUSTOM REQUIRED.**

### K58: QSA sparse attention

- **Category / priority:** Attention / P2.

- **Usage:** `attention/backends/qsa.py` consumes K57 selections.
- **Semantics:** QSA-specific block-sparse attention over its paged value/cache
  layout, causal/variable-length batches and exact scaling.
- **Backends:** custom CUDA CuTe; ROCm/NPU TBD. No provider semantics verified.
- **Performance:** QSA dominant kernel; highest architecture/shape risk.
  **Status: CUSTOM REQUIRED.**

### K59: Paged MHA KV-cache write

- **Category / priority:** Cache / P0.

- **Usage:** `MhaTokenToKVPool.set_kv_buffer` writes fresh K/V before paged
  extend/decode.
- **Semantics:** scatter packed `[T,Hkv,D]` K and V to logical `[P,S,Hkv,D]`
  slots from int32 locations, with no overwrite outside selected slots.
- **Backends:** custom CUDA/CuTe; ROCm Triton; NPU TBD (current Ascend attention
  does not fuse writes). Provider fused store layouts are optional variants.
- **Performance:** every layer/token; mandatory P0 and bandwidth-bound; fuse
  K20 only after ABI proof. **Status: CUSTOM REQUIRED.**

### K60: MLA KV-cache read/write

- **Category / priority:** Cache / P2.

- **Usage:** `MlaTokenToKVPool` get/set methods and MLA backends.
- **Semantics:** gather/scatter compressed latent plus rotary fields to paged
  MLA slots; support page-table transfer and model-specific latent widths.
- **Backends:** custom CUDA; ROCm Triton; NPU TBD. FlashInfer attention consumes
  the cache but does not own TokenSpeed's complete lifecycle operation.
- **Performance:** every MLA token/layer; keep provider cache layout native.
  **Status: CUSTOM REQUIRED.**

### K61: Quantized KV write + scale layout

- **Category / priority:** Cache / P1.

- **Usage:** FP8/MXFP8 MHA/MLA/DSA cache classes write payload and associated
  scales.
- **Semantics:** quantize or copy FP8 bytes and write scale rows at matching page
  slots; scale granularity/order is a format tag, not implicit metadata.
- **Backends:** custom CUDA/CuTe; ROCm Triton; NPU TBD. FlashInfer K75/K76 may
  provide quantization, not full cache scatter.
- **Performance:** each quantized attention layer/token; fuse K59/K60.
  **Status: CUSTOM REQUIRED.**

### K62: State-page verify/copy/commit

- **Category / priority:** Cache / P1.

- **Usage:** hybrid KDA/GDN cache classes and `draft_page_staging.py` during
  speculative staging and acceptance.
- **Semantics:** copy selected state rows/pages between committed and draft
  buffers, commit accepted prefix only, handle overlapping/self copies safely.
- **Backends:** custom CUDA; ROCm portable kernels; NPU TBD. Providers do not
  manage runtime ownership.
- **Performance:** every speculative batch; bandwidth/ordering critical.
  **Status: CUSTOM REQUIRED.**

### K63: Host/device cache-range transfer

- **Category / priority:** Cache / P1.

- **Usage:** cache transfer recipes and L2/EPD cache movement for offload or
  remote prefill.
- **Semantics:** asynchronous typed byte-range copies of specified layers/pages,
  respecting stream/event lifetime and pinned host buffers.
- **Backends:** CUDA runtime memcpy/streams; ROCm HIP memcpy/streams; NPU TBD.
  Specialist kernel providers are unnecessary.
- **Performance:** cache hit/migration only; overlap with compute and batch
  adjacent ranges. **Status: TODO.**

### K64: Cache-group table unpack/decode locations

- **Category / priority:** Cache / P1.

- **Usage:** `cache_groups.py` and KV-cache recipes translate grouped logical
  allocations for heterogeneous attention layers.
- **Semantics:** unpack cache-group/page tables to per-layer/per-token int32
  locations, preserving invalid sentinels and speculative offsets.
- **Backends:** custom CUDA/CUB; ROCm Triton/rocPRIM; NPU TBD.
- **Performance:** each heterogeneous batch setup; capture-friendly and fused
  with K12 where practical. **Status: CUSTOM REQUIRED.**

### K65: Metadata tape/sequence-index/tau generation

- **Category / priority:** Metadata / P0.

- **Usage:** `cache_metadata.py`, `ops/metadata.PrepTape`,
  `seq_idx_from_cu_seqlens`, and Inkling `log_scaling_tau` generate attention
  metadata, token-to-sequence IDs, offsets and position-dependent tau.
- **Semantics:** deterministic int32 scans/fills from batch lengths plus FP32
  `1+alpha*log(max((position+1)/n_floor,1))`; outputs directly match the
  attention/cache ABI and graph-static capacities.
- **Backends:** custom CUDA+CUB; ROCm Triton/rocPRIM; NPU TBD.
- **CUDA (committed):** `metadata.h:BuildSequenceIndexAsync` (gap-aware binary search) and `LogScalingTauAsync`; the attention-tape variant remains.
- **Performance:** every P0 batch/step; small-launch sensitive. **Status: PARTIAL (seq-index + tau committed; tape pending).**

### K66: Generic device fill/copy/zero

- **Category / priority:** Memory / P0.

- **Usage:** cache initialization, page invalidation, graph buffers, masks and
  runtime state throughout execution.
- **Semantics:** typed or bytewise fill/copy at exact ranges, async on caller
  stream; includes zeroing counters/workspaces.
- **Backends:** backend-neutral `kernels/include/inferx/kernels/memory.h`;
  CUDA runtime plus `cuda/memory/device_memory.cu`; ROCm HIP plus
  `rocm/memory/device_memory.hip`; Ascend AscendCL plus
  `ascend/memory/device_memory.cc`. Host/device and peer cache transfers
  remain K63. The AscendCL API exposes only bytewise memset, so non-repeated
  typed fills return `kUnsupported` there until an AscendC pattern kernel is
  integrated; the same rejection-not-coercion rule applies to any backend
  without a matching primitive.
- **Performance:** ubiquitous; batch ranges and avoid hidden synchronization.
  Both CUDA and ROCm implementations allocate nothing, use the caller stream,
  and are graph capturable; the Ascend implementation is allocation-free and
  stream-ordered but has no graph-capture path in AscendCL. **Status: NEEDS
  VERIFICATION.** CUDA compiles and its device test (fill/zero/copy plus graph
  replay) passes, but needs broader shape coverage; ROCm still needs a HIP
  build/runner; Ascend compiles and links against a stub CANN but needs a real
  CANN toolchain build and device qualification.

### K67: Softmax top-k MoE routing

- **Category / priority:** MoE / P2.

- **Usage:** `layers/moe/topk.py` calls `moe_softmax_topk` for standard/Qwen
  routers.
- **Semantics:** FP32-stable softmax over `[T,E]`, exact top-k expert IDs and
  normalized weights, optional renormalization.
- **Backends:** custom CUDA+CUB; ROCm Gluon/Triton; NPU TBD. HPC fused MoE assumes
  preprocessed routing and does not replace this general policy.
- **Performance:** once per MoE layer; fuse softmax/select. **Status: CUSTOM
  REQUIRED.**

### K68: Specialized sigmoid/bias/group/sink MoE routing

- **Category / priority:** MoE / P2.

- **Usage:** `moe_sigmoid_bias_topk`, `inkling_topk`, and
  `minimax_biased_grouped_topk` implement DeepSeek/Inkling/MiniMax variants.
- **Semantics:** sigmoid scores plus correction bias, group-limited selection,
  shared-expert sink weights where configured, top-k IDs/weights and exact
  route/global normalization; FP32 ranking.
- **Backends:** custom CUDA+CUB; ROCm Gluon/Triton; NPU TBD. Generic provider
  top-k is insufficient.
- **Performance:** once per MoE layer; routing correctness affects model output.
  **Status: CUSTOM REQUIRED.**

### K69: DSV4 sqrt-softplus/hash routing

- **Category / priority:** MoE / P2.

- **Usage:** DeepSeek V4 model and MoE ops select its specialized router.
- **Semantics:** exact sqrt/softplus scoring and hash/group policy defined by
  DSV4, producing expert IDs, weights and dispatch metadata.
- **Backends:** custom CUDA/CUB; ROCm AMD Gluon; NPU TBD. No audited CUDA
  provider exposes it.
- **Performance:** every DSV4 MoE layer; fuse statistics/select. **Status:
  CUSTOM REQUIRED.**

### K70: Expert token permute/count/sort

- **Category / priority:** MoE / P2.

- **Usage:** `moe_plan` and `FusedMoE` prepare expert-contiguous tokens, counts
  and reverse maps.
- **Semantics:** stable or explicitly unstable sort by expert, prefix offsets,
  optional EP ownership filtering, and reversible token/top-k mapping.
- **Backends:** custom CUDA+CUB; ROCm Gluon/rocPRIM; NPU TBD. HPC metadata can be
  consumed only inside its K71--K73 adapter.
- **Performance:** every MoE layer; avoid standalone gather with fused K71.
  **Status: CUSTOM REQUIRED.**

### K71: MoE grouped GEMM

- **Category / priority:** MoE / P2.

- **Usage:** `moe_apply` executes expert gate/up and down projections using K70
  counts/offsets.
- **Semantics:** variable-M grouped GEMMs over E experts, BF16/FP8/FP4 weights
  and exact scale layouts; zero-token experts valid.
- **Backends:** CUDA HPC-Ops FP8 grouped GEMM **USE WITH ADAPTER**, with
  FlashInfer/CUTLASS variants for BF16/FP4; ROCm Gluon/CK; NPU TBD.
- **Performance:** dominant MoE compute; model/shape tuning required. **Status:
  ADAPTER REQUIRED.**

### K72: Expert activation/requantization

- **Category / priority:** MoE / P2.

- **Usage:** `moe_apply` transforms gate/up output before expert down GEMM.
- **Semantics:** model-selected K17/K18 formula plus optional FP8/FP4 activation
  requantization and per-block scales in expert-contiguous layout.
- **Backends:** CUDA HPC-Ops activation path **USE WITH ADAPTER** for compatible
  FP8 formulas, FlashInfer fused-MoE fallback for qualified BF16/FP4/FP8 paths,
  then custom; ROCm Gluon; NPU TBD.
- **Performance:** every MoE expert pass; fuse between K71 variants. **Status:
  ADAPTER REQUIRED.**

### K73: Token unpermute/weighted reduce/finalize shared

- **Category / priority:** MoE / P2.

- **Usage:** MoE finalize scatters expert outputs and combines top-k/shared
  expert results.
- **Semantics:** reverse K70 mapping, multiply router weights, FP32-accurate sum
  over selected experts, add shared output when present.
- **Backends:** CUDA HPC-Ops fused MoE **USE WITH ADAPTER** for qualified FP8
  path, FlashInfer fused-MoE fallback for qualified BF16/FP4/FP8 paths, then
  custom CUDA; ROCm Gluon/Triton; NPU TBD.
- **Performance:** every MoE layer; fuse down GEMM epilogue. **Status: ADAPTER
  REQUIRED.**

### K74: Latent-MoE projections/fused tail

- **Category / priority:** MoE / P2.

- **Usage:** `layers/moe/latent.py` and Kimi K3/DSV4 latent expert paths.
- **Semantics:** model-specific latent projections, expert computation and tail
  combine using its packed weights/metadata; not ordinary grouped GEMM alone.
- **Backends:** custom CUDA CUTLASS/CuTe; ROCm AMD Gluon; NPU TBD. Providers do
  not expose the complete latent contract.
- **Performance:** dominant applicable MoE path; high fusion/shape risk.
  **Status: CUSTOM REQUIRED.**

### K75: FP8 quantize/dequantize/scales

- **Category / priority:** Quantization / P1.

- **Usage:** FP8 dense, MoE and KV paths in `layers/quantization/fp8.py` and
  `tk/ops/quantization`.
- **Semantics:** format-tagged E4M3/E5M2 conversion with per-tensor/row/block
  scale compute or supplied scale, saturation and dequantization.
- **Backends:** custom CUDA/CuTe; ROCm Triton/hipBLASLt utilities; NPU TBD.
  HPC-Ops exposes only coupled subsets, while FlashInfer's public standalone
  quantizers do not cover TokenSpeed's complete ordinary FP8 scale contract.
- **Performance:** every FP8 layer; fuse into producer/consumer. **Status:
  CUSTOM REQUIRED.**

### K76: MXFP8 quantize/dequantize

- **Category / priority:** Quantization / P2.

- **Usage:** MXFP8 cache and expert paths use dedicated microscale ops.
- **Semantics:** MXFP8 element bytes plus exponent/scale blocks in the exact
  block orientation required by GEMM/cache consumer.
- **Backends:** CUDA FlashInfer SM100+; ROCm Triton where hardware format is
  supported; NPU TBD. HPC-Ops has no general MXFP8 API.
- **Performance:** additional quantized modes; layout conformance dominates.
  **Status: NEEDS VERIFICATION.**

### K77: NVFP4 quantize/dequantize

- **Category / priority:** Quantization / P2.

- **Usage:** `layers/quantization/nvfp4.py` for Blackwell dense/MoE weights and
  activations.
- **Semantics:** NVFP4 nibble packing, dual-level scales and saturation exactly
  as consumer GEMM expects.
- **Backends:** CUDA FlashInfer Blackwell; ROCm custom conversion/fallback (not
  a native interchangeable format); NPU TBD.
- **Performance:** every NVFP4 layer; fuse K22 and GEMM. **Status: NEEDS
  VERIFICATION.**

### K78: MXFP4 quantize/dequantize

- **Category / priority:** Quantization / P2.

- **Usage:** `layers/quantization/mxfp4.py` and GPT-OSS/quantized MoE paths.
- **Semantics:** microscaled FP4 block packing/scales with exact exponent layout
  and optional FP8 activation interface.
- **Backends:** CUDA FlashInfer Blackwell; ROCm Gluon/CK native MXFP4 path; NPU
  TBD.
- **Performance:** every MXFP4 expert/linear; layout tests required. **Status:
  NEEDS VERIFICATION.**

### K79: GPTQ/Marlin weight repack

- **Category / priority:** Quantization / P2.

- **Usage:** compressed-tensor WNa16 and `gptq_marlin_moe.py` load/execute packed
  weights.
- **Semantics:** permute/repack quantized weights, scales, zeros and group map to
  Marlin tile layout; deterministic offline/runtime conversion.
- **Backends:** existing Marlin CUDA; custom ROCm format-specific path; NPU TBD.
  HPC-Ops/FlashInfer do not own this exact loader contract.
- **Performance:** model load plus W4 execution; prevent hot-path repacking.
  **Status: NEEDS VERIFICATION.**

### K80: Logit softcap + grammar mask

- **Category / priority:** Logits / P1.

- **Usage:** `LogitsProcessor` applies model softcap; grammar manager applies
  allowed-token mask before sampling.
- **Semantics:** optional `cap*tanh(logit/cap)` then set disallowed vocabulary
  entries to `-inf`, over sharded/full `[B,V]` FP logits.
- **Backends:** custom CUDA elementwise; ROCm Triton; NPU TBD. Sampling providers
  do not own grammar-state masks.
- **Performance:** each generation step when enabled; fuse K81. **Status: CUSTOM
  REQUIRED.**

### K81: Logits penalties/bias

- **Category / priority:** Sampling / P1.

- **Usage:** Triton/full sampling backends apply temperature, logit bias,
  repetition, frequency and presence penalties.
- **Semantics:** exact per-request parameters and token-history counts over
  `[B,V]`; penalty sign behavior and order match TokenSpeed.
- **Backends:** custom CUDA/CUB fused pipeline; ROCm Triton; NPU TBD. HPC sampler
  lacks the full policy and FlashInfer does not replace all modifiers.
- **Performance:** every non-greedy step; fuse K80/K82--K87. **Status: CUSTOM
  REQUIRED.**

### K82: Token-count update

- **Category / priority:** Sampling / P1.

- **Usage:** sampling backends build/update per-request token occurrence counts
  for K81.
- **Semantics:** scatter-add sampled/history token IDs into `[B,V]` dense or
  compact count representation, reset finished/new requests correctly.
- **Backends:** custom CUDA/CUB; ROCm Triton; NPU TBD. HPC penalty mask is not the
  full count contract.
- **Performance:** each sampled token; prefer compact touched-token state.
  **Status: CUSTOM REQUIRED.**

### K83: Softmax/log-softmax

- **Category / priority:** Sampling / P1.

- **Usage:** sampling probabilities and selected/full logprob output.
- **Semantics:** stable rowwise softmax/log-softmax over `[B,V]`, masking
  `-inf`, FP32 accumulation, FP16/BF16/FP32 input where registered.
- **Backends:** CUDA FlashInfer sampling softmax; ROCm Triton; NPU TBD.
- **Performance:** each sampling/logprob step; avoid materialization when fused
  with selection. **Status: NEEDS VERIFICATION.**

### K84: Top-k

- **Category / priority:** Sampling / P1.

- **Usage:** sampling backends filter vocabulary; distinct from sparse/MoE
  top-k because its input/output contract is probabilities/logits.
- **Semantics:** per-row exact k largest, deterministic tie policy and mask/
  compact output compatible with subsequent sampling.
- **Backends:** CUDA FlashInfer; ROCm Gluon/Triton; NPU TBD. HPC sampler is not
  primary because the full policy is incomplete.
- **Performance:** each configured step; sorting-free implementation.
  **Status: NEEDS VERIFICATION.**

### K85: Top-p

- **Category / priority:** Sampling / P1.

- **Usage:** sampling backends perform nucleus filtering.
- **Semantics:** retain smallest descending-probability prefix whose cumulative
  mass reaches per-request `p`, with TokenSpeed boundary/tie behavior.
- **Backends:** CUDA FlashInfer; ROCm Triton; NPU TBD.
- **Performance:** each configured step; sorting-free algorithm requires
  distributional equivalence tests. **Status: NEEDS VERIFICATION.**

### K86: Min-p

- **Category / priority:** Sampling / P1.

- **Usage:** sampling backends filter probabilities relative to row maximum.
- **Semantics:** retain tokens meeting TokenSpeed's per-request min-p threshold,
  with fallback ensuring a valid sample.
- **Backends:** CUDA FlashInfer; ROCm Triton; NPU TBD. HPC sampler lacks min-p.
- **Performance:** each configured step; fuse K83/K87. **Status: NEEDS
  VERIFICATION.**

### K87: Stochastic/Gumbel sampling

- **Category / priority:** Sampling / P1.

- **Usage:** FlashInfer/Triton sampling backends draw final token after filters;
  Gumbel variants support deterministic distributed/speculative flows.
- **Semantics:** sample from filtered row using caller RNG seed/offset, return
  token ID and validity; deterministic replay is part of ABI.
- **Backends:** CUDA FlashInfer plus RNG adapter; ROCm Triton; NPU TBD.
- **Performance:** every stochastic step; seed/capture/concurrency tests are
  mandatory. **Status: NEEDS VERIFICATION.**

### K88: Greedy argmax (local)

- **Category / priority:** Sampling / P0.

- **Usage:** `sampling/backends/greedy.py` selects local maximum when the logits
  vector is not TP-distributed.
- **Semantics:** argmax `[B,V] -> [B]` with specified first-index tie behavior
  and NaN policy.
- **Backends:** CUDA CUB/ATen; ROCm rocPRIM/ATen; NPU TBD. No specialist
  dependency needed.
- **Performance:** every P0 decode step; tiny-latency reduction. **Status:
  TODO.**

### K89: Selected/top logprobs

- **Category / priority:** Sampling / P1.

- **Usage:** `LogitsProcessor` and `engine/logprobs.py` gather prompt/output
  token logprobs and optional top-N alternatives.
- **Semantics:** stable logsumexp normalization, indexed gather and top-N over
  local/full vocab, with TP offsets and prompt-position mapping.
- **Backends:** custom CUDA/CUB; ROCm Triton; NPU TBD. FlashInfer pieces do not
  expose the whole TokenSpeed result structure.
- **Performance:** request-option path; fuse K83/K84 and avoid full copies.
  **Status: CUSTOM REQUIRED.**

### K90: Speculative chain verify/sample

- **Category / priority:** Sampling / P2.

- **Usage:** draft verification path calls chain speculative sampling for
  proposed token trees/chains.
- **Semantics:** accept/reject proposals from target/draft probabilities with
  deterministic RNG, sample correction token and return accepted length.
- **Backends:** CUDA FlashInfer; ROCm custom Triton/HIP; NPU TBD. State commit is
  separately K62/K51/K55.
- **Performance:** each speculative step; probabilistic conformance is critical.
  **Status: NEEDS VERIFICATION.**

### K91: All-reduce

- **Category / priority:** Communication / P1.

- **Usage:** `distributed/comm_ops.py` and TP linear/model paths reduce partial
  hidden vectors across ranks.
- **Semantics:** in-place/out-of-place sum over identical typed tensors in a
  communicator, caller stream ordering and graph capture explicit.
- **Backends:** CUDA NCCL; ROCm RCCL; Ascend HCCL. HPC-Ops allreduce is fused
  and topology-specific, not the general operation.
- **Performance:** multiple times per TP layer; overlap/fusion opportunity with
  K13/K19. **Status: TODO.**

### K92: All-gather

- **Category / priority:** Communication / P1.

- **Usage:** TP vocabulary/hidden assembly in distributed layers and logits.
- **Semantics:** concatenate equal or supported variable shards in rank order,
  preserving dtype and caller stream.
- **Backends:** CUDA NCCL; ROCm RCCL; Ascend HCCL.
- **Performance:** model/sampling dependent; avoid when sharded consumer exists.
  **Status: TODO.**

### K93: Reduce-scatter

- **Category / priority:** Communication / P1.

- **Usage:** distributed communication backend for sharded hidden results.
- **Semantics:** reduce sum then distribute rank-ordered equal shards; async and
  graph semantics match K91.
- **Backends:** CUDA NCCL; ROCm RCCL; Ascend HCCL.
- **Performance:** TP hot path where selected; pair with producer GEMM.
  **Status: TODO.**

### K94: All-to-all expert dispatch

- **Category / priority:** Communication / P2.

- **Usage:** expert-parallel MoE dispatch/return, including optional DeepEP
  route.
- **Semantics:** variable token counts per rank/expert, exchange payload plus
  inverse routing metadata, then restore original ordering.
- **Backends:** CUDA NCCL or optional DeepEP; ROCm RCCL all-to-all; NPU TBD.
  Ascend HCCL's checked wrapper is even-split only and does not close the
  variable expert-dispatch contract; HPC-Ops has no general EP transport.
- **Performance:** each EP MoE layer; topology/count skew is high risk.
  **Status: TODO.**

### K95: Send/recv/broadcast

- **Category / priority:** Communication / P1.

- **Usage:** pipeline-parallel activation transfer and distributed model/
  scheduler state synchronization.
- **Semantics:** point-to-point and root broadcast of exact buffers with group,
  rank, stream and lifetime specified.
- **Backends:** CUDA NCCL; ROCm RCCL; Ascend HCCL send/recv and process-group
  broadcast.
- **Performance:** PP/control dependent; batch messages and overlap.
  **Status: TODO.**

### K96: Fused collective+residual+norm

- **Category / priority:** Communication / P1.

- **Usage:** fused allreduce backends replace K91+K19+K13 for eligible TP hidden
  tensors.
- **Semantics:** mathematically identical sum across ranks, residual add and
  RMSNorm; return residual/normalized outputs as unfused path would.
- **Backends:** CUDA HPC-Ops **USE WITH ADAPTER** on qualified single-node NVLink
  BF16 shapes; ROCm custom RCCL/HIP; NPU TBD.
- **Performance:** per eligible TP layer, P1; topology/workspace/barrier safety
  is a major risk. **Status: ADAPTER REQUIRED.**

### K97: Distributed argmax

- **Category / priority:** Communication / P1.

- **Usage:** TP greedy sampling when vocabulary logits remain sharded.
- **Semantics:** reduce `(value, global_vocab_id)` pairs across ranks with exact
  tie/NaN policy, return one ID per sequence without full gather.
- **Backends:** custom CUDA+NCCL; custom HIP+RCCL; NPU TBD. General allreduce is
  insufficient without pair/tie semantics.
- **Performance:** every TP greedy step; latency critical. **Status: CUSTOM
  REQUIRED.**

### K98: DP sampling gather/swap

- **Category / priority:** Communication / P1.

- **Usage:** `dp_sampling_comm.py`/`dp_sampling_swap.py` exchange variable active
  rows and place sampled results back on owning DP ranks.
- **Semantics:** gather compact logits/metadata, maintain owner permutation,
  then inverse-scatter token/results; graph-static buffers allowed.
- **Backends:** custom CUDA+NCCL; ROCm HIP+RCCL; NPU TBD. No provider API matches
  runtime ownership semantics.
- **Performance:** every DP sampled step; small-message launch/collective risk.
  **Status: CUSTOM REQUIRED.**

### K99: Causal/depthwise Conv1D + recurrent state

- **Category / priority:** State/conv / P2.

- **Usage:** `attention/linear/causal_conv1d.py` in GDN/KDA hybrid layers.
- **Semantics:** channelwise causal convolution over prompt chunks or one-token
  rolling windows, with explicit persisted convolution state.
- **Backends:** custom CUDA/CuTe (cuDNN fallback for bulk); ROCm Triton/MIOpen;
  NPU TBD. Provider attention APIs do not own state lifecycle.
- **Performance:** each hybrid layer/token; decode fusion into K54.
  **Status: CUSTOM REQUIRED.**

### K100: PLE n-gram hash

- **Category / priority:** PLE / P2.

- **Usage:** `layers/qwen4_exp_ple.py` hashes token n-grams for Qwen4 PLE.
- **Semantics:** reproduce configured integer rolling hash, bucket mapping and
  padding/sequence-boundary behavior from token IDs.
- **Backends:** custom CUDA; ROCm Triton; NPU TBD. No provider equivalent.
- **Performance:** each Qwen4 PLE layer/batch; memory/launch bound. **Status:
  CUSTOM REQUIRED.**

### K101: PLE page gather/scatter

- **Category / priority:** PLE / P2.

- **Usage:** Qwen4 PLE gathers hashed embeddings/state from and scatters updates
  to its paged cache.
- **Semantics:** indexed paged rows with valid masks, duplicate policy and
  reversible sequence ordering.
- **Backends:** custom CUDA/CUB; ROCm Triton; NPU TBD.
- **Performance:** each PLE step; fuse K100/K102 where possible. **Status:
  CUSTOM REQUIRED.**

### K102: PLE dilated convolution

- **Category / priority:** PLE / P2.

- **Usage:** Qwen4 PLE applies configured dilated causal convolution over the
  gathered n-gram representation.
- **Semantics:** exact dilation/kernel width and sequence-boundary padding over
  `[T,C]`, with paged history on decode.
- **Backends:** custom CUDA/CuTe; ROCm Triton; NPU TBD. cuDNN alone does not own
  the paged decode contract.
- **Performance:** Qwen4 PLE hot path; specialize prompt/decode. **Status: CUSTOM
  REQUIRED.**

### K103: mHC pre/post Sinkhorn mapping

- **Category / priority:** Residual / P2.

- **Usage:** `layers/hyperconnection.py` in GLM/Kimi residual streams.
- **Semantics:** compute Sinkhorn-normalized connection matrices, map multiple
  residual streams before/after block, exact normalization iterations/eps.
- **Backends:** custom CUDA/CuTe; ROCm AMD Gluon/Triton; NPU TBD. HPC `iHC` is
  **NOT SUITABLE** without semantic/layout equivalence.
- **Performance:** each applicable block; small-matrix fusion and numerical
  stability matter. **Status: CUSTOM REQUIRED.**

### K104: Gated residual hyperconnection

- **Category / priority:** Residual / P2.

- **Usage:** `layers/hyperconnection.py` applies model gates to multiple residual
  streams and recombines them.
- **Semantics:** configured linear/gated map over `[T,n_stream,D]`, preserving
  stream order and residual dtype.
- **Backends:** custom CUDA/CuTe; ROCm Triton; NPU TBD. No verified provider.
- **Performance:** each applicable block; fuse gate/map/combine. **Status:
  CUSTOM REQUIRED.**

### K105: AttnRes

- **Category / priority:** Residual / P2.

- **Usage:** MiniMax M3 residual-attention path calls TokenSpeed AttnRes ops.
- **Semantics:** model-specific attention-weighted residual stream selection/
  merge, including its normalization and indexing contract.
- **Backends:** custom CUDA/CuTe; ROCm Gluon/Triton; NPU TBD. Generic attention
  and HPC iHC do not match.
- **Performance:** each MiniMax block; specialized small reductions. **Status:
  CUSTOM REQUIRED.**

### K106: Multimodal Conv2D/Conv3D patch projection

- **Category / priority:** Multimodal / P2.

- **Usage:** Qwen/Kimi/GLM vision and audio components create spatial/temporal
  patch embeddings.
- **Semantics:** configured Conv2D/3D stride/padding/groups over NCHW/NCDHW
  FP16/BF16 tensors, returning model token ordering.
- **Backends:** CUDA cuDNN; ROCm MIOpen; NPU TBD. No specialist provider needed.
- **Performance:** once per multimodal input/encoder stage; layout conversion can
  dominate. **Status: TODO.**

### K107: Multimodal pooling/interpolation/patch merge

- **Category / priority:** Multimodal / P2.

- **Usage:** conditional model encoders and `multimodal_runtime.py` resize,
  pool, merge and scatter media tokens into the text stream.
- **Semantics:** exact interpolation alignment, pooling/window order and
  gather/scatter indices; FP16/BF16 payload.
- **Backends:** CUDA cuDNN/ATen; ROCm MIOpen/ATen; NPU TBD.
- **Performance:** input-dependent, usually amortized; correctness of spatial/
  temporal ordering is primary. **Status: TODO.**

### K108: DFlash2 grouped dynamic convolution/transition

- **Category / priority:** Draft model / P2.

- **Usage:** `models/dflash2.py` and `execution/drafter/dflash2.py` generate
  draft hidden transitions/candidates.
- **Semantics:** grouped, input-dependent convolution/einsum and candidate-state
  transition in the exact DFlash2 group/channel ordering.
- **Backends:** CUDA ATen/cuDNN baseline; ROCm MIOpen/ATen; NPU TBD. Optimize
  with custom CuTe only after profiling.
- **Performance:** every DFlash2 draft step; candidate width and small-batch
  latency are risks. **Status: TODO.**

## Backend-neutral interface

Use a compile-time-selected backend table plus typed operation descriptors, not
a virtual tensor framework. InferX model/runtime code sees only stable
`inferx::kernels` types derived from the audited TokenSpeed contracts; backend
adapters retain the metadata that determines kernel legality.

```text
TokenSpeed execution contract
        |
        v
InferX runtime/model adapter
        |
        v
inferx::kernels (typed descriptors + capability query)
        |
        +-- cuda  -> HPC-Ops | FlashInfer | CUTLASS/CuTe/CUDA | NCCL
        +-- rocm  -> hipBLASLt/CK | Triton/Gluon/HIP | RCCL
        `-- ascend -> AscendCL/aclnn | HCCL (torch_npu plugin as evidence)
```

Representative C++ shape (names illustrative):

```cpp
struct PagedAttentionDesc {
  DType q_dtype, kv_dtype, out_dtype;
  CacheLayout cache_layout;       // includes format/scales
  int q_heads, kv_heads, qk_dim, value_dim, page_size, max_q_len;
  bool causal, return_lse;
  Window window;
  Span<const int32_t> page_table, kv_lengths, qo_offsets;
};

Status mha_decode(const PagedAttentionDesc&, TensorView q,
                  TensorView k_cache, TensorView v_cache, TensorView out,
                  Stream, Workspace);
Capability query_mha_decode(const PagedAttentionDesc&);
```

Rules:

- Dispatch once per batch/plan, outside the inner tile loop.  The selected
  function pointer and provider plan/workspace are cached in an attention/MoE/
  sampling plan object; no per-token C++ virtual call or tensor boxing.
- `TensorView` is a non-owning pointer/dtype/rank/sizes/strides view.  Operation
  descriptors carry page layout, scale format, head geometry, GQA ratio,
  window/sinks/logit-cap, determinism and architecture requirements explicitly.
- A provider adapter may permute metadata during plan/build, but must not make a
  hidden cache copy on every decode.  Cache layout is selected when the pool is
  created.
- Capability results are `SUPPORTED`, `SUPPORTED_WITH_ADAPTER`, or a concrete
  rejection reason.  Fallback dispatch follows HPC-Ops -> FlashInfer -> custom
  for CUDA without weakening semantics.
- Communication accepts an opaque backend communicator and caller stream; it
  is not expressed as a tensor operator.  Cache/state mutations declare alias
  and ordering rules.

## Proposed source layout

Create a directory only when its first implementation lands.  This is the
target layout, not a request to commit empty placeholders:

```text
kernels/
├── include/inferx/kernels/            # InferX-owned neutral API
├── common/                            # validation, dispatch, format enums
├── cuda/
│   ├── attention/                     # MHA, MLA, relative, DSA/DSV4/MSA/QSA
│   ├── gemm/                          # dense, routed, projection, grouped
│   ├── norm/                          # norms and residual fusions
│   ├── rope/
│   ├── activation/
│   ├── kv_cache/                      # writes, locations, state commit
│   ├── moe/
│   ├── quantization/
│   ├── sampling/
│   ├── communication/
│   ├── memory/
│   ├── linear_attention/              # GDN/KDA/causal conv
│   ├── ple/
│   └── multimodal/
├── rocm/
│   ├── attention/
│   ├── gemm/
│   ├── norm/
│   ├── activation/
│   ├── kv_cache/
│   ├── moe/
│   ├── quantization/
│   ├── sampling/
│   ├── communication/
│   ├── memory/
│   ├── linear_attention/
│   └── ple/
├── ascend/
│   ├── attention/                     # only after Ascend kernel integration
│   ├── communication/                 # HCCL wrapper
│   ├── memory/                        # K66 fill/copy via AscendCL
│   ├── norm/
│   └── rope/
└── tests/                             # neutral and backend qualification
```

For example, `cuda::PagedMhaDecode` owns the public wrapper while its selected
plan calls FlashInfer; `cuda::MoeGroupedGemm` can call HPC-Ops for FP8 and
FlashInfer/CUTLASS for other formats. InferX model/runtime code and any
TokenSpeed compatibility adapter depend only on `inferx::kernels`, never on
those providers directly.

## Dependency policy

| Dependency | Kernels used | Why / optionality | Compile-time / runtime | Architecture and dtype notes |
|---|---|---|---|---|
| CUTLASS/CuTe | K03 and custom GEMM/attention/fusions | **Required CUDA build dependency** once CUDA kernels land; header/DSL implementation detail | build-time headers/DSL; generated objects at runtime, no Python | InferX already pins CUTLASS; TokenSpeed requires `nvidia-cutlass-dsl[cu13]==4.7.1`; emit separate SM targets; FP16/BF16/FP8/FP4 availability depends on SM. |
| HPC-Ops | K08, K24, K71--K73, K96 | **Optional CUDA provider**, experimental until qualification; valuable but only six primary owners | compile/link optional adapter; matching per-SM HPC extension at runtime | pinned source advertises CUDA >=12.9 and builds SM90/100/103; BF16/FP8, specialized shapes/topology. |
| FlashInfer | Primary K04--K06, K13--K14, K17, K20--K21, K25--K26, K32--K35, K40--K41, K48--K50, K76--K78, K83--K87, K90; fallback K24, K71--K73 | **Optional CUDA provider with a required custom fallback for P0**; broadest reusable inference coverage | prefer native C++/linked kernels; do not introduce Python/JIT into the C++ runtime; TokenSpeed currently pins `flashinfer-python==0.6.18` | pinned upstream spans SM75--SM121, but individual kernels are narrower; FP4/MXFP8 and TRT-LLM paths need architecture gates. |
| cuBLASLt/cuDNN/CUDA/CUB | K01--K03, K09--K11, K23, K63, K66, K88, K106--K108 and custom support | **Required CUDA toolkit components** | compile/link and driver runtime; cuBLAS is dynamic-loaded on first GEMM after CUDA warm-up (linking it directly runs libcublasLt's ELF initializers first, which breaks device enumeration on WSL2/CUDA 12.0) | toolkit/driver matrix must be pinned; dtype support follows GPU/toolkit. |
| Marlin | K05 W4A16 execution and K79 repack | **Optional CUDA provider**, enabled only for GPTQ/Marlin checkpoints | compile/link CUDA extension and packed weights at runtime | exact integer width, group size, zeros/scales and supported SM must be a capability key. |
| NCCL | K91--K98 | **Optional unless TP/PP/EP/DP enabled** | link or dynamic-load backend; NCCL runtime | topology and graph-capture qualification; BF16/FP16/FP32/byte payloads. |
| DeepEP | optimized K94 | **Optional only for expert parallelism**; NCCL remains correctness transport | separate CUDA package/extension and runtime communication resources | qualify NVIDIA SM, NVLink/RDMA topology, token-count limits and FP8/BF16 payload layout. |
| hipBLASLt/rocBLAS/MIOpen/rocPRIM | ROCm dense, conv, reduction baseline | **Required ROCm backend components** | ROCm compile/runtime | qualify per gfx target and data format. |
| TokenSpeed-Kernel-AMD / Triton-Gluon / CK | 96 ROCm reusable rows, especially attention/MoE | **Optional optimized ROCm provider**; portable correctness fallback remains | TokenSpeed requires `tokenspeed-kernel-amd>=0.1.3`; the Python package is evidence/source today, while production C++ integration needs an explicit packaging decision | audited package targets gfx950/gfx1250 subsets; BF16, FP8 and MXFP4 coverage varies. |
| RCCL | ROCm K91--K98 | **Optional unless distributed enabled** | ROCm link/runtime | qualify collectives/topology separately from NCCL. |
| NPU vendor plugin / CANN | K66 plus future Ascend rows | **CANN/AscendCL is required when `INFERX_ENABLE_ASCEND=ON`**; the `torch_npu` evidence package stays an optional plugin outside core runtime | compile/link `libascendcl` via `INFERX_ASCEND_CANN_PATH`/`ASCEND_TOOLKIT_HOME`; qualify per Ascend SoC | current proven evidence beyond K66 is Ascend `torch_npu` plus HCCL: norm/RoPE, FP16/BF16 restricted attention, and non-expert communication; aclnn/AscendC kernel strategy for the remaining rows is still TBD. |

Recommendation: keep HPC-Ops and FlashInfer optional providers, both disabled
unless their capability probe and ABI-version check pass.  FlashInfer is the
more important coverage dependency, but the runtime should still own a P0
fallback so its Python/JIT packaging is not mandatory.  HPC-Ops should graduate
from experimental only after SM90/100/103 build, license/transitive dependency,
numerical, graph, topology, and representative-shape benchmarks pass.
TokenSpeed's currently pinned DeepGEMM, FlashMLA, FlashAttention 3/4,
TensorRT-LLM-kernel, FlashKDA, CuTeDSL-KDA, FHT and Iris packages should not all
become dependencies of the new layer: their algorithms/tests are useful
reference material, but K04/K24/K32--K34/K52--K54/K94 already have the selected
provider or custom ownership above. Add one only if an architecture-specific
benchmark demonstrates a capability or material performance gap, and keep it
behind the same optional adapter ABI.

## Verification gates

Every implementation must add:

1. a descriptor/capability test that accepts every supported tuple and rejects
   nearby unsupported dtype/layout/stride/head/page/GQA/window combination;
2. numerical comparison against TokenSpeed's reference/PyTorch implementation
   for prefill, chunked extend, decode, mixed lengths, empty rows, MTP and cache
   wrap/page boundaries where applicable;
3. cache byte-layout and scale-layout golden tests, including provider adapters;
4. deterministic/tie/RNG tests for routing and sampling;
5. graph-capture, non-default-stream, aliasing and workspace lifetime tests;
6. per-architecture benchmarks with a correctness fallback threshold.  An API
   existing upstream earns `AVAILABLE`, not `DONE`; `DONE` requires these gates
   in this repository.

## Implementation roadmap

### Stage 1 - Minimal CUDA Llama BF16 execution (P0)

Implement the exact 18-operation P0 closure in dependency order:

1. K66 memory primitives; K09 layout; K10 gather/scatter; K11 scans; K12/K65
   batch/page metadata. **Status: committed for CUDA (K65 tape variant and
   multi-block scans pending).**
2. K01 embedding, K03 dense GEMM, K13 RMSNorm, K19 residual add, K17 SwiGLU,
   and K20 RoPE. **Status: committed for CUDA (K19 gate variants and K20 mRoPE
   pending).**
3. K59 KV write, K24 packed prefill, K25 paged extend and K26 paged decode.
4. K02 LM head and K88 greedy argmax.
5. Run `LlamaForCausalLM` single-GPU BF16 prompt -> decode -> token output,
   including multi-request mixed lengths and page boundaries.  HPC-Ops K24 and
   FlashInfer K13/K17/K20/K25/K26 are optional accelerators, not correctness
   blockers.

### Stage 2 - Production batching and sampling (P1)

1. K61--K64 quantized/state cache lifecycle and transfer.
2. K80--K89 full logits processing, penalties, top-k/top-p/min-p, RNG and
   logprobs; fuse after independent conformance.
3. K04/K06/K07/K35/K75 production FP8 and split-attention paths.
4. K91--K98 TP/DP communication; first unfused NCCL, then qualified K96.

### Stage 3 - Additional dense/MLA and quantized models (P2)

1. K14--K16, K18, K21--K23 and K27--K35.
2. K05 and K76--K79, with format-tagged checkpoint/load tests.
3. K90 speculative chain verification plus K62 commit integration.

### Stage 4 - Sparse attention, MoE and hybrid state models (P2/P3)

1. K38--K47 DSA/DSV4 index/cache/attention pipeline.
2. K67--K74 complete MoE routing -> dispatch -> grouped GEMM -> finalize.
3. K48--K55 GDN/KDA plus K99 stateful convolution; K51/K55 only after base
   speculative correctness.
4. K56--K58 QSA, then K36--K37 MSA.

### Stage 5 - Specialized and multimodal models (P2)

1. K100--K102 Qwen4 PLE.
2. K103--K105 hyperconnection/AttnRes.
3. K106--K108 multimodal and DFlash2 paths.

### Stage 6 - Additional backends

1. Bring ROCm to the Stage-1 P0 contract using hipBLASLt/Triton-Gluon/RCCL,
   then qualify the 95 reusable rows and implement its ten gaps.
2. Integrate the ten proven Ascend/HCCL operations behind the NPU plugin ABI.
3. The NPU runtime is fixed to Ascend (CANN/AscendCL, `kernels/ascend/`);
   select the model subset, aclnn/AscendC kernel strategy, tensor formats,
   and quantization before assigning the other 97 rows; then implement its
   P0 closure and only afterward performance fusions.

## Highest-risk kernels

- K26, K34, K41, K47, K58: paged/sparse decode combines strict cache ABI,
  irregular lengths, GQA and per-token latency.
- K43--K47 and K56--K58: DSV4/QSA have model-specific indexing and no complete
  reusable CUDA provider.
- K36--K37: MSA is architecture-specialized and lacks a reusable ROCm path.
- K52--K55/K99: KDA recurrence, convolution state and speculative commit must be
  transactionally correct.
- K69--K74/K94: routing skew, grouped low-precision GEMM, inverse permutation
  and expert-parallel transport interact.
- K05/K76--K79: four-bit/microscale formats are architecture- and byte-layout-
  specific; a wrong “compatible” format silently corrupts results.
- K87/K90: distributed/speculative RNG equivalence and deterministic replay.
- K96: fused communication uses topology-specific memory ordering/barriers.

## Final answers

1. **Logical kernel count:** 108, derived from registered model execution,
   runtime/cache/sampling/distributed paths, kernel registry calls, and reachable
   PyTorch tensor operations.
2. **Priority split:** P0 18, P1 25, P2 63, P3 2.
3. **CUDA:** HPC-Ops direct 0; HPC-Ops adapter 6; FlashInfer primary 25;
   standard/already-present CUDA provider 15; custom CUTLASS/CuTe/CUDA 62.
4. **ROCm:** 95 have a reusable correctness path, 10 need custom work, and 3
   QSA operations remain TBD. Major gaps are MSA, NVIDIA-specific FP4/Marlin,
   high-accuracy router GEMM, speculative verification, fused communication and
   production tuning outside the package's gfx targets.
5. **NPU:** 10 proven reusable Ascend/HCCL operations, 1 InferX-owned row
   committed (K66 memory primitives against AscendCL, pending CANN/device
   qualification), 97 TBD pending the model-subset/format decisions now that
   the runtime is fixed to Ascend.
   Quantization, MoE, specialized attention/state models, sampling, cache
   lifecycle, expert transport and communication fusions remain unknown.
6. **Minimum P0 CUDA set:** K01--K03, K09--K13, K17, K19--K20, K24--K26,
   K59, K65--K66 and K88 (18 operations) for single-GPU BF16 Llama greedy
   prefill/extend/decode.
7. **Highest performance risk:** paged/sparse decode, DSV4/QSA/MSA, recurrent
   KDA state, grouped low-precision MoE, microscale four-bit formats, and fused
   communication.
8. **External dependencies:** CUTLASS/CuTe and standard CUDA components are the
   CUDA foundation; FlashInfer and HPC-Ops remain optional capability-gated
   providers; NCCL/RCCL are optional with distributed execution; use ROCm's
   native stack and AMD package only in the ROCm backend; keep NPU runtime a
   plugin.
9. **First implementation:** land the backend-neutral descriptors and Stage-1
   P0 metadata/memory primitives, then GEMM/norm/activation/RoPE, then cache and
   attention, and finally LM-head/argmax and the Llama end-to-end test.
