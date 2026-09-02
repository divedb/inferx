#!/usr/bin/env python3
"""Shared workload matrix for the InferX-vs-TokenSpeed benchmark.

Both the C++ harness (benchmarks/kernels/kernels_benchmark.cc) and the
TokenSpeed runner (tools/bench/run_tokenspeed_bench.py) consume this matrix
so the comparison uses identical shapes, dtypes, and seeds.

Each entry: (op, workload_name, params dict).
"""

# dtype keys: "fp32" | "fp16" | "bf16"
WORKLOADS = [
    # activation: fused silu_and_mul [T, 2D] -> [T, D]
    ("act_mul_silu", "T4096_D5120", {"tokens": 4096, "dim": 5120, "kind": "silu", "dtype": "bf16"}),
    ("act_mul_silu", "T16384_D8192", {"tokens": 16384, "dim": 8192, "kind": "silu", "dtype": "bf16"}),
    ("act_mul_gelu", "T4096_D5120", {"tokens": 4096, "dim": 5120, "kind": "gelu", "dtype": "bf16"}),
    # layernorm
    ("rmsnorm", "T4096_H5120", {"tokens": 4096, "hidden": 5120, "dtype": "bf16"}),
    ("fused_add_rmsnorm", "T4096_H5120", {"tokens": 4096, "hidden": 5120, "dtype": "bf16"}),
    ("qk_rmsnorm", "T4096_H32_D128", {"tokens": 4096, "heads": 32, "kv_heads": 8, "head_dim": 128, "dtype": "bf16"}),
    # rope (TokenSpeed embedding category)
    ("rope", "T4096_H32_D128", {"tokens": 4096, "heads": 32, "kv_heads": 8, "head_dim": 128, "dtype": "bf16"}),
    # gemm
    ("gemm", "M4096_K4096_N4096", {"tokens": 4096, "in": 4096, "out": 4096, "dtype": "bf16"}),
    ("gemm", "M8_K4096_N4096", {"tokens": 8, "in": 4096, "out": 4096, "dtype": "bf16"}),
    # attention decode: B sequences, S context, H32/KV8/D128 (bf16 only)
    ("attention_decode", "B1_S4096", {"batch": 1, "context": 4096, "heads": 32, "kv_heads": 8, "head_dim": 128}),
    ("attention_decode", "B8_S2048", {"batch": 8, "context": 2048, "heads": 32, "kv_heads": 8, "head_dim": 128}),
    # sampling
    ("argmax", "B8_V131072", {"rows": 8, "vocab": 131072, "dtype": "fp32"}),
    ("top_p_renorm", "B8_V131072", {"rows": 8, "vocab": 131072, "top_p": 0.95, "dtype": "fp32"}),
    # quantization
    ("fp8_quant", "T4096_D7168", {"tokens": 4096, "dim": 7168, "group": 128, "dtype": "fp32"}),
    # moe routing
    ("softmax_topk", "T512_E256_K8", {"tokens": 512, "experts": 256, "top_k": 8, "dtype": "fp32"}),
    # transform
    ("hadamard", "T4096_D128", {"tokens": 4096, "dim": 128, "dtype": "bf16"}),
]

if __name__ == "__main__":
    for op, name, params in WORKLOADS:
        print(f"{op:20s} {name:20s} {params}")
