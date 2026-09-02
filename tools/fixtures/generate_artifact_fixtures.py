#!/usr/bin/env python3
"""Generate deterministic, deliberately tiny artifact fixtures."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "model"


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def safetensors_file(header: str, payload: bytes) -> bytes:
    encoded = header.encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def packed_f16_bytes(shape: list[int]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * 2


def tiny_llama_files() -> dict[Path, bytes]:
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2,
        "intermediate_size": 4,
        "max_position_embeddings": 16,
        "model_type": "llama",
        "num_attention_heads": 1,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 0.00001,
        "tie_word_embeddings": False,
        "vocab_size": 4,
    }
    tensors = [
        ("lm_head.weight", [4, 2]),
        ("model.embed_tokens.weight", [4, 2]),
        ("model.layers.0.input_layernorm.weight", [2]),
        ("model.layers.0.mlp.down_proj.weight", [2, 4]),
        ("model.layers.0.mlp.gate_proj.weight", [4, 2]),
        ("model.layers.0.mlp.up_proj.weight", [4, 2]),
        ("model.layers.0.post_attention_layernorm.weight", [2]),
        ("model.layers.0.self_attn.k_proj.weight", [2, 2]),
        ("model.layers.0.self_attn.o_proj.weight", [2, 2]),
        ("model.layers.0.self_attn.q_proj.weight", [2, 2]),
        ("model.layers.0.self_attn.v_proj.weight", [2, 2]),
        ("model.norm.weight", [2]),
    ]
    shard_names = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    shard_tensors: list[list[tuple[str, list[int], int]]] = [[], []]
    weight_map: dict[str, str] = {}
    total_size = 0
    for parameter_id, (name, shape) in enumerate(tensors):
        shard = parameter_id % len(shard_names)
        shard_tensors[shard].append((name, shape, parameter_id + 1))
        weight_map[name] = shard_names[shard]
        total_size += packed_f16_bytes(shape)

    generated: dict[Path, bytes] = {
        Path("tiny_llama/config.json"): json_bytes(config),
        Path("tiny_llama/model.safetensors.index.json"): json_bytes(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        ),
    }
    for shard_name, entries in zip(shard_names, shard_tensors, strict=True):
        offset = 0
        header: dict[str, Any] = {}
        payload = bytearray()
        for name, shape, fill in entries:
            size = packed_f16_bytes(shape)
            header[name] = {
                "data_offsets": [offset, offset + size],
                "dtype": "F16",
                "shape": shape,
            }
            payload.extend(bytes([fill]) * size)
            offset += size
        header_text = json.dumps(header, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        generated[Path("tiny_llama") / shard_name] = safetensors_file(
            header_text, bytes(payload)
        )
    return generated


def corpus_files() -> dict[Path, bytes]:
    cases: list[tuple[str, bytes, bool, str]] = []

    def add(name: str, header: str, payload: bytes, accepted: bool, reason: str) -> None:
        cases.append((name, safetensors_file(header, payload), accepted, reason))

    add(
        "valid_scalar.safetensors",
        '{"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}',
        bytes([1, 2, 3, 4]),
        True,
        "scalar shape covers the payload",
    )
    add(
        "valid_empty.safetensors",
        '{"empty":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}}',
        b"",
        True,
        "zero-element tensor at a payload boundary",
    )
    add(
        "valid_subbyte.safetensors",
        '{"packed":{"dtype":"F4","shape":[3],"data_offsets":[0,2]}}',
        bytes([0x12, 0x30]),
        True,
        "sub-byte dtype rounds up to complete bytes",
    )
    add(
        "invalid_overlap.safetensors",
        '{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},'
        '"b":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}',
        bytes([1, 2]),
        False,
        "overlapping tensor ranges",
    )
    add(
        "invalid_hole.safetensors",
        '{"a":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}',
        bytes([0, 1, 2]),
        False,
        "unclaimed byte before tensor payload",
    )
    add(
        "invalid_bad_dtype.safetensors",
        '{"a":{"dtype":"F128","shape":[1],"data_offsets":[0,1]}}',
        bytes([0]),
        False,
        "unknown dtype",
    )
    add(
        "invalid_duplicate_name.safetensors",
        '{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
        '"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}',
        bytes([1, 2]),
        False,
        "duplicate tensor name",
    )
    cases.append(
        (
            "invalid_truncated_header.safetensors",
            struct.pack("<Q", 128) + b"{}",
            False,
            "declared header exceeds file",
        )
    )

    generated: dict[Path, bytes] = {}
    table = ["file\taccepted\treason"]
    for name, contents, accepted, reason in cases:
        generated[Path("safetensors") / name] = contents
        table.append(f"{name}\t{str(accepted).lower()}\t{reason}")
    generated[Path("safetensors/corpus.tsv")] = ("\n".join(table) + "\n").encode("utf-8")
    return generated


def expected_files() -> dict[Path, bytes]:
    readme = b"""# Artifact fixtures

These files are intentionally tiny, deterministic parser and model-package fixtures.
Regenerate them with `python3 tools/fixtures/generate_artifact_fixtures.py`.

The safetensors corpus records InferX's accepted/rejected boundary contract. It is
not a substitute for an independent differential check against the official Rust
implementation.
"""
    return {
        Path("README.md"): readme,
        **tiny_llama_files(),
        **corpus_files(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail if committed fixtures differ"
    )
    args = parser.parse_args()

    failures: list[str] = []
    for relative, expected in expected_files().items():
        path = FIXTURE_ROOT / relative
        if args.check:
            if not path.is_file() or path.read_bytes() != expected:
                failures.append(str(path.relative_to(ROOT)))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)

    if failures:
        print("artifact fixtures are stale:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
