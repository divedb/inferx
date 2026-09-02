# Kernel implementation TODO

Status against the TokenSpeed reference inventory
(`tokenspeed-kernel/python/tokenspeed_kernel/ops`), following the ADR 0031
provider chain hpc-ops → FlashInfer → custom. Completed operators and the
benchmark results live in `docs/benchmarks/RESULTS.md`; this file tracks
what remains.

## Not implemented at all

### Attention families (largest gap; model-specific layouts)

Model-specific attention variants with no SM89 backend at the pinned
revisions (hpc-ops is SM90+; FlashInfer's pinned MLA/sparse kernels are
Hopper-oriented). Implement when target models require them.

- [ ] MLA: `mla_project_value`, `mla_normalize_project_query`,
      `mla_prefill`, `mla_extend_with_kvcache`, `mla_decode_with_kvcache`
- [ ] DSA (DeepSeek sparse attention): `dsa_plan`, `dsa_prefill`,
      `dsa_decode`, `dsa_prefill_topk`, `dsa_decode_topk`
- [ ] DSV4 family: `dsv4_prefill`, `dsv4_decode`, top-k variants,
      `dsv4_swa_cache_insert`, `dsv4_csa_indexer_fp8_cache_insert`,
      `dsv4_indexer_cache_format`, `dsv4_padded_heads`
- [ ] MSA (MiniMax sparse): `msa_decode_with_kvcache`,
      `msa_extend_with_kvcache`, `minimax_indexer`
- [ ] GDN (gated deltanet linear attention): `gdn_chunk_prefill`,
      `gdn_decode_step`, `gdn_decode_mtp`, `gdn_replay_commit`
- [ ] KDA (Kimi delta attention): `kda_paged_prefill`, `kda_paged_decode`,
      `kda_fused_paged_decode`, `kda_fused_paged_verify`, replay
- [ ] Relative MHA (shearing bias): `rel_mha_plan`, `rel_mha_prefill`,
      `rel_mha_extend_with_kvcache`, `rel_mha_decode_with_kvcache`
- [ ] `attn_merge_state` (LSE-based partial-output merge)
- [ ] MXFP8 block-scaled decode path (UE8M0 per-32 scale factors)

### MoE beyond routing

Routing (`softmax_topk`, `sigmoid_bias_topk`) is done and oracle-verified;
the apply side is not. Needs grouped GEMM + a dispatch/combine strategy
(DeepEP or equivalent) — record a new ADR before starting.

- [ ] `moe_apply` — bf16 unquant
- [ ] `moe_apply` — block-FP8
- [ ] `moe_apply` — NVFP4 / MXFP4 / MXINT4
- [ ] DeepEP dispatch/combine (a2a legs) + `deepep_scatter`/`deepep_gather`
- [ ] `dsv4_select_experts` (sqrt-softplus router, optional hash routing)
- [ ] DSV4 MegaMoE (single fused SM100 kernel — SM90+ hardware required)
- [ ] Latent MoE projections (`latent_moe_input_projections`,
      `latent_moe_expert_shared`)

### KV-cache category (entirely scaffold today)

Needed for real serving; partially blocked on the attention families above.

- [ ] `store_kv_cache` (standard paged K/V append)
- [ ] `set/get_mla_kv_buffer` (MLA compressed-cache write/read)
- [ ] Fused FP8 quantize + KV store (`fused_fp8_set_kv_buffer`)
- [ ] MXFP8 KV quantization (`quantize_store_kv_mxfp8`, interleaved SF
      layout)
- [ ] `copy_state_rows` (linear-attention state caches)
- [ ] `gather_page_table_with_padding`, `compute_group_decode_locs`,
      group-table unpacking
- [ ] Cross-layer KV transfers (multi-GPU; needs > 1 GPU to qualify)

### Communication (multi-GPU; blocked on single-GPU runner)

- [ ] `allreduce_residual_rmsnorm`
- [ ] `reducescatter_residual_rmsnorm`
- [ ] `allgather_dual_rmsnorm`
- [ ] `allreduce_lane_latent_norm`, `allreduce_fusion_lane`

### Sampling beyond renorm/argmax

`argmax`, `top_p_renorm`, `top_k_renorm` are done; the sampling kernels
themselves are not.

- [ ] Top-k / top-p / min-p sampling from probs/logits (Philox-backed)
- [ ] Gumbel pool-sampling family (speculative-decoding candidate pools)
- [ ] `apply_penalties_logit_bias_inplace`, `accumulate_counts_inplace`
- [ ] `selected_token_logprobs`
- [ ] Spec-decode verify chains (`verify_chain_target_sampled`,
      `verify_chain_greedy`)
- [ ] Distributed vocab-parallel argmax (NVLink symmetric memory)
- [ ] Fused topk-topp (prepare/renorm; needs a precompiled CUDA wheel or
      owned equivalent)

### Quantization beyond FP8

- [ ] MXFP8 encoder (FP8 data + encoded vector scales)
- [ ] NVFP4 encoder (packed E2M1x2 + E4M3 scale factors, linear/swizzled)
- [ ] MXFP4 encoder (packed E2M1x2 + UE8M0 scales)

### GEMM variants

- [ ] Skinny-M GEMV path for decode shapes (`M8` regression: 0.17× vs
      cuBLASLt in RESULTS.md) — highest-priority item in this file
- [ ] Block-FP8 `mm`/`bmm` (`fp8_linear`, prepared variants)
- [ ] MXFP4 / NVFP4 `mm`
- [ ] Grouped GEMM (also a MoE-apply prerequisite)
- [ ] Kimi-K3 composite projections (latent/QKV-gate/router/shared)

### Miscellaneous operators

- [ ] `situ_and_mul` (SiTU activation variant)
- [ ] `rmsnorm_gated_sigmoid` (gated RMSNorm)
- [ ] `swiglu_oai` (OpenAI-style SwiGLU)
- [ ] Fused activation + FP8/NVFP4 quant (`prepare_fp8_linear_activation`,
      `silu_and_mul_fuse_block_quant`, `silu_and_mul_fuse_nvfp4_quant`)
- [ ] `grouped_gemma_rmsnorm` (per-group (1+w) multiplier; the whole-row
      `gemma_rmsnorm` is done)
- [ ] Metadata prep-tape (CUDA-graph replay metadata interpreter — infra,
      not a math kernel)

## Implemented but restricted

Lift these envelope restrictions as the need arises; each is enforced in
the provider `Available()` probe and documented in the category header.

| op | current restriction | notes |
|---|---|---|
| attention decode/prefill | head dims {64, 128}; FP16/BF16 only; no sliding window / logit cap / sinks / LSE | FlashInfer pinned-kernel limits; widening needs more template instantiations and variant plumbing |
| rope | head dims {32, 64, 128, 256} | compile-time template dims |
| top_p_renorm | FP32 only | pinned kernel's reduction helpers fail to instantiate for 16-bit |
| fp8 quant | FP32 inputs only | widen to 16-bit inputs |
| rmsnorm family | hidden multiple of vector width (8 for 16-bit) | tail loop needed for odd hidden |
| MoE routing | experts ≤ 4096 | shared-memory row; needs tiling beyond |
| mhc / attn_res | ≤ 8 residual streams / ≤ 13 candidates | fixed shared-memory bounds |
| gemm | no skinny-M/GEMV path | see GEMM variants above |
| hadamard | last dim == 128 | matches TokenSpeed's Triton kernel |
| softmax/sigmoid topk | returns ids in selection order only | no stable-tie option beyond lowest-index |

## Suggested priority

1. Skinny-GEMV — fixes the worst measured regression.
2. KV-cache category — required for serving.
3. MLA/DSA families — required for DeepSeek-class models.
4. Sampling kernels + MoE apply (needs grouped GEMM first).
5. Envelope widening from the table above.
