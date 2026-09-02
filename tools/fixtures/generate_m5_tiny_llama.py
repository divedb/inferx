#!/usr/bin/env python3
"""Generate the M5 EOS tiny-Llama fixture.

Derives `tests/fixtures/model/tiny_llama_eos` from the M3 artifact fixture
`tests/fixtures/model/tiny_llama` by rewriting weight values in place: every
tensor keeps its exact name, shape, dtype (F16), shard, and byte range, so
the safetensors index stays valid and only the data bytes change.

The construction is deterministic and hand-reasoned:

- embeddings are positive vectors, so the RMSNormed final hidden state has
  strictly positive entries for every token;
- every attention/MLP projection is zero, so the residual stream is exactly
  the (normalized) embedding;
- the LM head maps every positive hidden state to a dominant logit for
  token 3, the tokenizer's `<|endoftext|>`.

Greedy generation therefore emits exactly [3] (EOS) for any prompt, which
exercises the real stop path (FinishReason::kEos, kStopMatched) end to end.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "tests/fixtures/model/tiny_llama"
TARGET = REPO / "tests/fixtures/model/tiny_llama_eos"

EMBED_ROWS = [(1.0, 2.0), (2.0, 2.0), (1.0, 1.0), (3.0, 1.0)]
NORM_VALUE = (1.0, 1.0)
LM_HEAD_EOS = (10.0, 10.0)
LM_HEAD_OTHER = (-10.0, -10.0)
EOS_TOKEN = 3


def f16_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<e", value) for value in values)


def replacement(name: str, shape: list[int]) -> bytes | None:
    """Replacement data for `name`, or None to keep the original bytes."""
    if name == "model.embed_tokens.weight":
        return f16_bytes([value for row in EMBED_ROWS for value in row])
    if name in ("model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.norm.weight"):
        return f16_bytes(list(NORM_VALUE))
    if name == "lm_head.weight":
        rows = [LM_HEAD_EOS if token == EOS_TOKEN else LM_HEAD_OTHER for token in range(4)]
        return f16_bytes([value for row in rows for value in row])
    # q/k/v/o projections and the MLP gate/up/down projections: zeros.
    if name.startswith("model.layers.0."):
        return f16_bytes([0.0] * (shape[0] * shape[1]))
    return None


def rewrite(shard: Path, destination: Path) -> None:
    payload = shard.read_bytes()
    header_size = struct.unpack_from("<Q", payload, 0)[0]
    header = json.loads(payload[8 : 8 + header_size])
    data_start = 8 + header_size
    mutated = bytearray(payload)
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        data = replacement(name, tensor["shape"])
        if data is None:
            continue
        begin = data_start + tensor["data_offsets"][0]
        assert len(data) == tensor["data_offsets"][1] - tensor["data_offsets"][0], name
        mutated[begin : begin + len(data)] = data
    destination.write_bytes(mutated)


def patch_config(source: Path, destination: Path) -> None:
    config = json.loads(source.read_text())
    config["eos_token_id"] = EOS_TOKEN
    destination.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the fixture is current instead of writing")
    args = parser.parse_args()

    if args.check:
        for shard in sorted(SOURCE.glob("*.safetensors")):
            expected = bytearray(shard.read_bytes())
            rewrite_into = bytearray()
            # Re-derive in memory and compare.
            candidate = TARGET / shard.name
            if not candidate.exists():
                print(f"missing: {candidate}")
                return 1
            actual = candidate.read_bytes()
            probe = bytearray(expected)
            header_size = struct.unpack_from("<Q", probe, 0)[0]
            header = json.loads(probe[8 : 8 + header_size])
            data_start = 8 + header_size
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                data = replacement(name, tensor["shape"])
                if data is None:
                    continue
                begin = data_start + tensor["data_offsets"][0]
                probe[begin : begin + len(data)] = data
            del rewrite_into
            if bytes(probe) != actual:
                print(f"stale: {candidate}")
                return 1
        print("tiny_llama_eos fixture is current")
        return 0

    TARGET.mkdir(parents=True, exist_ok=True)
    for shard in sorted(SOURCE.glob("*.safetensors")):
        rewrite(shard, TARGET / shard.name)
    patch_config(SOURCE / "config.json", TARGET / "config.json")
    for support in ("model.safetensors.index.json", "tokenizer.json"):
        shutil.copy2(SOURCE / support, TARGET / support)
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
