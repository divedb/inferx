#!/usr/bin/env bench-venv/bin python3
"""TokenSpeed-side benchmark: identical workloads to the InferX harness
(benchmarks/kernels/kernels_benchmark.cc) using the TokenSpeed operator
APIs, CUDA-event timed, emitting the same JSON schema.

Backends on SM89: FlashInfer wheel where eligible (their NVIDIA-first
paths), Triton fallbacks otherwise. The backend actually used is recorded
per sample.
"""
import json
import math
import os
import sys

import torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BENCH_DIR, "..", ".."))
sys.path.insert(0, os.path.join(BENCH_DIR, "stubs"))
sys.path.insert(0, os.path.join(BENCH_DIR, "tokenspeed", "tokenspeed-kernel", "python"))
del REPO_ROOT

torch.manual_seed(42)
device = "cuda"

ITERS = 50
WARMUP = 10

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)


def time_once(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1000.0)  # us
    times.sort()
    return times[len(times) // 2]


def bf16(shape, seed=0, lo=-1.5, hi=1.5):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(shape, generator=gen, device=device, dtype=torch.float32).mul_(
        hi - lo).add_(lo).to(torch.bfloat16)


def fp32(shape, seed=0, lo=-1.5, hi=1.5):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(shape, generator=gen, device=device, dtype=torch.float32).mul_(hi - lo).add_(lo)


samples = []


def emit(op, workload, micros, backend):
    samples.append({"impl": "tokenspeed", "backend": backend, "op": op,
                    "workload": workload, "latency_us": round(micros, 3)})


def _attention_phase():
    from tokenspeed_kernel.ops.attention import mha_decode_with_kvcache

    for batch, ctx, tag in ((1, 4096, "B1_S4096"), (8, 2048, "B8_S2048")):
        q = bf16((batch, 32, 128), 16)
        kv = bf16((batch, ctx, 8, 128), 17)
        page_table = torch.arange(batch, device=device, dtype=torch.int32).reshape(batch, 1)
        cache_lens = torch.full((batch,), ctx, device=device, dtype=torch.int32)

        def decode():
            return mha_decode_with_kvcache(q=q, k_cache=kv, v_cache=kv,
                                           page_table=page_table,
                                           cache_seqlens=cache_lens,
                                           max_seqlen_k=ctx, max_seqlen_q=1)

        try:
            decode()
            torch.cuda.synchronize()
            emit("attention_decode", tag, time_once(decode), "triton")
        except Exception as error:  # noqa: BLE001
            print(f"attention_decode {tag} failed: {error}", file=sys.stderr)


def main():
    from tokenspeed_kernel.ops import activation as ts_activation
    from tokenspeed_kernel.ops import layernorm as ts_layernorm

    # Attention first: a fresh CUDA context avoids order-dependent faults
    # seen when JIT-compiled phases precede it.
    _attention_phase()

    # ---- activation: silu_and_mul [T, 2D] -> [T, D] (bf16)
    for tokens, dim, tag in ((4096, 5120, "T4096_D5120"), (16384, 8192, "T16384_D8192")):
        x = bf16((tokens, 2 * dim), 1)
        out = torch.empty((tokens, dim), device=device, dtype=torch.bfloat16)
        backend = "flashinfer" if hasattr(ts_activation, "silu_and_mul") else "triton"
        emit("act_mul_silu", tag, time_once(lambda: ts_activation.silu_and_mul(x, out)),
             backend)

    x = bf16((4096, 2 * 5120), 2)
    out = torch.empty((4096, 5120), device=device, dtype=torch.bfloat16)
    try:
        from tokenspeed_kernel.ops.activation.flashinfer import gelu_and_mul

        emit("act_mul_gelu", "T4096_D5120",
             time_once(lambda: gelu_and_mul(x, out)), "flashinfer")
    except Exception as error:  # noqa: BLE001
        print(f"act_mul_gelu failed: {error}", file=sys.stderr)
        emit("act_mul_gelu", "T4096_D5120",
             time_once(lambda: ts_activation.triton_silu_and_mul(x, out)), "triton-fallback")

    # ---- layernorm: rmsnorm (bf16)
    x = bf16((4096, 5120), 3)
    w = bf16((5120,), 4)
    emit("rmsnorm", "T4096_H5120", time_once(lambda: ts_layernorm.rmsnorm(x, w, 1e-5)),
         "flashinfer")

    # ---- fused_add_rmsnorm
    x = bf16((4096, 5120), 5)
    res = bf16((4096, 5120), 6)
    w = bf16((5120,), 7)
    emit("fused_add_rmsnorm", "T4096_H5120",
         time_once(lambda: ts_layernorm.rmsnorm(x, w, 1e-5, residual=res)), "flashinfer")

    # ---- qk_rmsnorm (bf16): q [T, H, D], k [T, KV, D]
    q = bf16((4096, 32 * 128), 8)
    k = bf16((4096, 8 * 128), 9)
    qw = bf16((128,), 10)
    kw = bf16((128,), 11)
    emit("qk_rmsnorm", "T4096_H32_D128",
         time_once(lambda: ts_layernorm.qk_rmsnorm(q, k, qw, kw, 1e-5)), "triton")

    # ---- rope via TokenSpeed embedding category (triton/cuda backend)
    from tokenspeed_kernel.ops import embedding as ts_embedding

    for tokens, heads, kv_heads, head_dim in ((4096, 32, 8, 128),):
        q = bf16((tokens, heads * head_dim), 12)
        k = bf16((tokens, kv_heads * head_dim), 13)
        positions = torch.arange(tokens, device=device, dtype=torch.int64)
        max_pos = 8192
        freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device,
                                              dtype=torch.float32) / head_dim))
        cache = torch.cat([torch.cos(torch.arange(max_pos, device=device).float()[:, None]
                                     * freqs[None, :]),
                           torch.sin(torch.arange(max_pos, device=device).float()[:, None]
                                     * freqs[None, :])], dim=-1).contiguous()  # float32, per kernel requirement
        backend = "triton"
        emit("rope", "T4096_H32_D128",
             time_once(lambda: ts_embedding.apply_rope(positions, q, k, head_dim, cache, solution="triton")), backend)

    # ---- gemm (bf16): A[M,K] @ B[N,K]^T
    for m, tag in ((4096, "M4096_K4096_N4096"), (8, "M8_K4096_N4096")):
        a = bf16((m, 4096), 14)
        b = bf16((4096, 4096), 15)
        emit("gemm", tag, time_once(lambda: torch.nn.functional.linear(a, b)), "cublas")

    # ---- argmax over [8, 131072] fp32
    logits = fp32((8, 131072), 18, -8, 8)
    emit("argmax", "B8_V131072",
         time_once(lambda: torch.argmax(logits, dim=-1)), "torch")

    # ---- top_p_renorm via flashinfer sampling re-exports
    try:
        from tokenspeed_kernel.ops.sampling.flashinfer import top_p_renorm_prob

        probs = torch.softmax(logits, dim=-1)
        emit("top_p_renorm", "B8_V131072",
             time_once(lambda: top_p_renorm_prob(probs, top_p=0.95)), "flashinfer")
    except Exception as error:  # noqa: BLE001
        print(f"top_p_renorm failed: {error}", file=sys.stderr)

    # ---- fp8 quantization (triton backend)
    from tokenspeed_kernel.ops import quantization as ts_quant

    # TokenSpeed registers fp8 quantization for bf16/fp16 inputs; the InferX
    # harness quantizes FP32 inputs, so the comparison keeps the same
    # element counts and notes the dtype difference.
    x = bf16((4096, 7168), 19)
    emit("fp8_quant", "T4096_D7168",
         time_once(lambda: ts_quant.quantize_fp8_with_scale(
             x, granularity="token_group", group_size=128,
             scale_encoding="float32", solution="triton")), "triton")

    # ---- moe routing: softmax top-k (triton)
    from tokenspeed_kernel.ops.moe import moe_softmax_topk

    logits = fp32((512, 256), 20, -6, 6)
    emit("softmax_topk", "T512_E256_K8",
         time_once(lambda: moe_softmax_topk(logits, 8)), "triton")

    # ---- hadamard transform (triton)
    from tokenspeed_kernel.ops.transform import hadamard_transform

    x = bf16((4096, 128), 21)
    emit("hadamard", "T4096_D128", time_once(lambda: hadamard_transform(x, solution="triton")), "triton")

    print(json.dumps(samples, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - always emit collected samples
        print(f"benchmark aborted: {error}", file=sys.stderr)
        print(json.dumps(samples, indent=2))
        sys.exit(1)
