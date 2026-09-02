#!/usr/bin/env python3
"""Run one pinned Llama checkpoint through InferX and vLLM and compare tokens."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "amakhov/tiny-random-llama"
DEFAULT_REVISION = "fbf68d33cf68a9d1d4b71b3d098ae82c8c14443b"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_TEMPERATURE = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inferx-binary", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    return parser.parse_args()


def run_inferx(args: argparse.Namespace) -> dict[str, object]:
    command = [
        str(args.inferx_binary),
        "run",
        "--model",
        args.model,
        "--revision",
        args.revision,
        "--download-dir",
        str(args.download_dir),
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def run_vllm(inferx: dict[str, object], args: argparse.Namespace) -> list[int]:
    # The V2 runner requires CUDA UVA, which is unavailable on some WSL hosts.
    # The established runner exercises the same model and sampling semantics.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    # Avoid an optional FlashInfer JIT dependency; native sampling performs
    # the same argmax operation at temperature zero.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError(
            "vLLM is required; install it in the Python environment running this test"
        ) from error

    prompt_token_ids = inferx["prompt_token_ids"]
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token, int) for token in prompt_token_ids
    ):
        raise RuntimeError("InferX returned invalid prompt_token_ids")

    llm = LLM(
        model=str(inferx["model_path"]),
        tokenizer=str(inferx["model_path"]),
        dtype="float32",
        enforce_eager=True,
        generation_config="vllm",
        gpu_memory_utilization=0.2,
        seed=0,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=0,
    )
    requests = [{"prompt_token_ids": prompt_token_ids}]
    outputs = llm.generate(requests, sampling, use_tqdm=False)
    return list(outputs[0].outputs[0].token_ids)


def main() -> int:
    args = parse_args()
    if args.temperature != DEFAULT_TEMPERATURE:
        raise ValueError("this deterministic differential test requires temperature 0.0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    inferx = run_inferx(args)
    vllm_tokens = run_vllm(inferx, args)
    inferx_tokens = inferx["output_token_ids"]
    if inferx_tokens != vllm_tokens:
        print(
            json.dumps(
                {
                    "match": False,
                    "temperature": args.temperature,
                    "prompt_token_ids": inferx["prompt_token_ids"],
                    "inferx_output_token_ids": inferx_tokens,
                    "vllm_output_token_ids": vllm_tokens,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "match": True,
                "model": inferx["model"],
                "revision": inferx["model_revision"],
                "temperature": args.temperature,
                "prompt_token_ids": inferx["prompt_token_ids"],
                "output_token_ids": inferx_tokens,
                "output_text": inferx["output_text"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
