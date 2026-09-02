#!/usr/bin/env python3
"""Plot the InferX-vs-TokenSpeed benchmark results (ADR 0032 addendum).

Inputs: inferx.json + tokenspeed.json (same schema; tokenspeed entries carry
a "backend" field). Outputs into docs/benchmarks/:
  - latency_comparison.png : grouped bars, log scale, per op x workload
  - speedup.png            : TokenSpeed/InferX ratio per workload
  - RESULTS.md             : summary table incl. backends and caveats
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OP_LABELS = {
    "act_mul_silu": "silu_and_mul",
    "act_mul_gelu": "gelu_and_mul",
    "rmsnorm": "rmsnorm",
    "fused_add_rmsnorm": "fused_add_rmsnorm",
    "qk_rmsnorm": "qk_rmsnorm",
    "rope": "rope",
    "gemm": "gemm",
    "attention_decode": "attn decode",
    "argmax": "argmax",
    "top_p_renorm": "top_p_renorm",
    "fp8_quant": "fp8 quant",
    "softmax_topk": "moe softmax-topk",
    "hadamard": "hadamard-128",
}


def load(path):
    with open(path) as handle:
        return json.load(handle)


def merge(inferx, tokenspeed):
    rows = []
    ts_index = {(row["op"], row["workload"]): row for row in tokenspeed}
    for row in inferx:
        key = (row["op"], row["workload"])
        counterpart = ts_index.pop(key, None)
        rows.append({
            "op": row["op"],
            "workload": row["workload"],
            "inferx": row["latency_us"],
            "tokenspeed": counterpart["latency_us"] if counterpart else None,
            "backend": counterpart["backend"] if counterpart else "-",
        })
    for key, row in ts_index.items():  # TokenSpeed-only workloads
        rows.append({"op": key[0], "workload": key[1], "inferx": None,
                     "tokenspeed": row["latency_us"], "backend": row["backend"]})
    return rows


def plot_latency(rows):
    labels, ours, theirs = [], [], []
    for row in rows:
        labels.append(f"{OP_LABELS.get(row['op'], row['op'])}\n{row['workload']}")
        ours.append(row["inferx"] if row["inferx"] else np.nan)
        theirs.append(row["tokenspeed"] if row["tokenspeed"] else np.nan)
    x = np.arange(len(labels))
    width = 0.38
    fig, axis = plt.subplots(figsize=(1.6 * len(labels), 6.5))
    axis.bar(x - width / 2, ours, width, label="InferX", color="#1f77b4")
    axis.bar(x + width / 2, theirs, width, label="TokenSpeed", color="#d62728")
    axis.set_yscale("log")
    axis.set_ylabel("kernel latency (us, median of 50, log scale)")
    axis.set_title("InferX vs TokenSpeed — same workloads, RTX 4080 SUPER (SM89)")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "latency_comparison.png", dpi=150)
    plt.close(fig)


def plot_speedup(rows):
    labels, ratios = [], []
    for row in rows:
        if row["inferx"] and row["tokenspeed"]:
            labels.append(f"{OP_LABELS.get(row['op'], row['op'])}\n{row['workload']}")
            ratios.append(row["tokenspeed"] / row["inferx"])
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(1.5 * len(labels), 5.5))
    colors = ["#2ca02c" if r >= 0.9 else "#ff7f0e" for r in ratios]
    axis.bar(x, ratios, color=colors)
    axis.axhline(1.0, color="k", linestyle="--", linewidth=1)
    axis.axhspan(0.9, 1.1, color="green", alpha=0.08)
    axis.set_ylabel("TokenSpeed / InferX latency (>1: InferX faster)")
    axis.set_title("Per-workload speedup (green: parity or better)")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "speedup.png", dpi=150)
    plt.close(fig)


def write_results(rows):
    lines = [
        "# InferX vs TokenSpeed kernel benchmark",
        "",
        "- Hardware: NVIDIA RTX 4080 SUPER (SM89), driver 591.86, CUDA 13.0.",
        "- Timing: CUDA events, median of 50 iterations after 10 warmups,",
        "  identical workload matrix (tools/bench/workloads.py).",
        "- InferX launches go through the unified dispatch surface with the",
        "  ADR 0031 provider chain active (hpc-ops -> flashinfer -> CUTLASS ->",
        "  owned); the provider actually used is listed for InferX too.",
        "- TokenSpeed backends per its own registry selection on this GPU.",
        "",
        "| operator | workload | InferX (us) | provider | TokenSpeed (us) | backend | ratio TS/IX |",
        "|---|---|---|---|---|---|---|",
    ]
    provider_by_op = {
        "act_mul_silu": "flashinfer", "act_mul_gelu": "flashinfer",
        "rmsnorm": "flashinfer", "fused_add_rmsnorm": "flashinfer",
        "qk_rmsnorm": "flashinfer", "rope": "flashinfer",
        "gemm": "cutlass", "attention_decode": "flashinfer",
        "argmax": "inferx_owned", "top_p_renorm": "flashinfer(fp32-only)",
        "fp8_quant": "inferx_owned", "softmax_topk": "inferx_owned",
        "hadamard": "inferx_owned",
    }
    for row in rows:
        ratio = (f"{row['tokenspeed'] / row['inferx']:.2f}"
                 if row["inferx"] and row["tokenspeed"] else "n/a")
        lines.append(
            f"| {OP_LABELS.get(row['op'], row['op'])} | {row['workload']} | "
            f"{row['inferx']:.1f} | {provider_by_op.get(row['op'], '-')} | "
            f"{row['tokenspeed']:.1f} | {row['backend']} | {ratio} |"
            if row["inferx"] and row["tokenspeed"] else
            f"| {OP_LABELS.get(row['op'], row['op'])} | {row['workload']} | "
            f"{'-' if not row['inferx'] else f'{row["inferx"]:.1f}'} | "
            f"{provider_by_op.get(row['op'], '-')} | "
            f"{'-' if not row['tokenspeed'] else f'{row["tokenspeed"]:.1f}'} | "
            f"{row['backend']} | {ratio} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- TokenSpeed `top_p_renorm`: the flashinfer wheel's sampling JIT module",
        "  does not compile under CUDA 13 (bundled CCCL removed `FlagHeads`);",
        "  no comparison sample was produced. InferX instantiates the same",
        "  kernel family ahead-of-time and covers it.",
        "- fp8_quant: InferX quantizes FP32 inputs; TokenSpeed registers bf16/",
        "  fp16 inputs only, so its sample uses bf16. Element counts match.",
        "- hadamard: TokenSpeed's preferred CUDA-wheel backend has no wheel for",
        "  this platform; the Triton fallback was forced (`solution=\"triton\"`).",
        "- deep_gemm / fast_hadamard_transform wheels could not be built here;",
        "  import-time stubs satisfy the registry and are never on the measured",
        "  paths.",
        "",
    ]
    (OUT_DIR / "RESULTS.md").write_text("\n".join(lines))


def main():
    inferx = load(sys.argv[1] if len(sys.argv) > 1 else "/tmp/bo.json")
    tokenspeed = load(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ts.json")
    rows = merge(inferx, tokenspeed)
    plot_latency(rows)
    plot_speedup(rows)
    write_results(rows)
    print(f"wrote {OUT_DIR}/latency_comparison.png, speedup.png, RESULTS.md "
          f"({len(rows)} workloads)")


if __name__ == "__main__":
    main()
