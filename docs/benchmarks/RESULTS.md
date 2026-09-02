# InferX vs TokenSpeed kernel benchmark

- Hardware: NVIDIA RTX 4080 SUPER (SM89), driver 591.86, CUDA 13.0.
- Timing: CUDA events, median of 50 iterations after 10 warmups,
  identical workload matrix (tools/bench/workloads.py).
- InferX launches go through the unified dispatch surface with the
  ADR 0031 provider chain active (hpc-ops -> flashinfer -> CUTLASS ->
  owned); the provider actually used is listed for InferX too.
- TokenSpeed backends per its own registry selection on this GPU.

| operator | workload | InferX (us) | provider | TokenSpeed (us) | backend | ratio TS/IX |
|---|---|---|---|---|---|---|
| silu_and_mul | T4096_D5120 | 196.6 | flashinfer | 197.6 | flashinfer | 1.01 |
| silu_and_mul | T16384_D8192 | 1260.2 | flashinfer | 1288.2 | flashinfer | 1.02 |
| gelu_and_mul | T4096_D5120 | 197.6 | flashinfer | 198.7 | flashinfer | 1.01 |
| rmsnorm | T4096_H5120 | 122.7 | flashinfer | 136.0 | flashinfer | 1.11 |
| fused_add_rmsnorm | T4096_H5120 | 270.3 | flashinfer | 280.4 | flashinfer | 1.04 |
| qk_rmsnorm | T4096_H32_D128 | 46.1 | flashinfer | 173.7 | triton | 3.77 |
| rope | T4096_H32_D128 | 123.5 | flashinfer | 114.7 | triton | 0.93 |
| gemm | M4096_K4096_N4096 | 1731.2 | cutlass | 1350.7 | cublas | 0.78 |
| gemm | M8_K4096_N4096 | 127.7 | cutlass | 21.5 | cublas | 0.17 |
| attn decode | B1_S4096 | 145.2 | flashinfer | 178.2 | triton | 1.23 |
| attn decode | B8_S2048 | 75.8 | flashinfer | 144.4 | triton | 1.91 |
| argmax | B8_V131072 | 71.3 | inferx_owned | 18.4 | torch | 0.26 |
| top_p_renorm | B8_V131072 | 42.6 | flashinfer(fp32-only) | - | - | n/a |
| fp8 quant | T4096_D7168 | 563.2 | inferx_owned | 393.2 | triton | 0.70 |
| moe softmax-topk | T512_E256_K8 | 30.4 | inferx_owned | 43.0 | triton | 1.41 |
| hadamard-128 | T4096_D128 | 33.8 | inferx_owned | 730.1 | triton | 21.61 |

## Caveats

- TokenSpeed `top_p_renorm`: the flashinfer wheel's sampling JIT module
  does not compile under CUDA 13 (bundled CCCL removed `FlagHeads`);
  no comparison sample was produced. InferX instantiates the same
  kernel family ahead-of-time and covers it.
- fp8_quant: InferX quantizes FP32 inputs; TokenSpeed registers bf16/
  fp16 inputs only, so its sample uses bf16. Element counts match.
- hadamard: TokenSpeed's preferred CUDA-wheel backend has no wheel for
  this platform; the Triton fallback was forced (`solution="triton"`).
- deep_gemm / fast_hadamard_transform wheels could not be built here;
  import-time stubs satisfy the registry and are never on the measured
  paths.
## NCU metric exports

Raw per-kernel CSV reports live under `docs/benchmarks/ncu/`
(`inferx.csv`, `inferx_attn.csv`, `tokenspeed.csv`, plus the original
`.ncu-rep` files for interactive inspection; collected with
`tools/bench/ncu_collect.sh`, CUDA 13 Nsight Compute 2025.3.1).

Representative medians for the attention-decode comparison:

| kernel | SM throughput | DRAM throughput | achieved occupancy |
|---|---|---|---|
| InferX — flashinfer `BatchDecodeWithPagedKVCacheKernel` | 3% | 14% | 8% |
| TokenSpeed — Triton decode (stage1/stage2) | 2% | 8% | 8% |

Both decode kernels run far below the memory-bandwidth ceiling at these
decode shapes; the flashinfer kernel's higher DRAM utilization matches its
1.2-1.9x latency advantage. NCU-replayed per-launch durations are not
directly comparable to the event-timed medians above (clock control and
replay differ), so use the CSVs for utilization/occupancy analysis and the
latency table for end-to-end timing.
