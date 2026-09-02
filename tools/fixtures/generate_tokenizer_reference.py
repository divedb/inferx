#!/usr/bin/env python3
"""Generate the tokenizer differential corpus.

For every checkpoint under tests/fixtures/model/tokenizer_reference/, produces
`cases.jsonl`: 2,500 deterministic (text, ids, decode) records produced by
the pinned Hugging Face Python `tokenizers` oracle. The committed corpus is
the acceptance evidence; this script regenerates it byte-identically from
its fixed seed.

Usage:
    out/fixtures-venv/bin/python tools/fixtures/generate_tokenizer_reference.py

The oracle version is pinned in tokenizer-requirements.lock and recorded in
MANIFEST.json next to the corpus.
"""

from __future__ import annotations

import json
import random
import sys
import zlib
from pathlib import Path

from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "tests/fixtures/model/tokenizer_reference"
CASES_PER_CHECKPOINT = 2500
SEED = 20260902

ENGLISH = [
    "The scheduler emits an immutable step plan.",
    "KV pages are reserved transactionally while planning.",
    "prefill and decode share one local engine initially",
    "TTFT, inter-token latency, and goodput at p99",
    "A successful submission means work was queued.",
    "the quick brown fox jumps over the lazy dog",
    "C++23 host code with CUDA C++ below the platform boundary",
    "Errors are typed and scoped to one request.",
    "  leading and trailing spaces  ",
    "multiple   internal    spaces",
    "tabs\there\tand\nnewlines\n\nagain",
    "punctuation: semicolons; dashes -- quotes 'both' \"kinds\" (parens)",
    "Mixed CASE and ALLCAPS and snake_case and kebab-case",
    "numbers 0 1 42 3.14159 -17 1e-9 0x1F 1_000_000",
    "https://example.com/path?query=1&other=a%20b",
    "def f(x): return x * x  # comment",
    '{"json": [1, 2, {"nested": null}], "ok": true}',
    "$ curl -sL --max-time 60 -o out.json https://hf.example/t.json",
]

MULTILINGUAL = [
    "一段中文文本，测试分词器的多字节处理能力。",
    "日本語のテキストです。ひらがな、カタカナ、漢字。",
    "한국어 텍스트 예시입니다. 한글 자모 조합.",
    "Русский текст для проверки кириллицы.",
    "Ελληνικό κείμενο για δοκιμή.",
    "نص عربي لاختبار الاتجاه من اليمين إلى اليسار.",
    "טקסט עברי לבדיקה.",
    "हिन्दी पाठ परीक्षण के लिए। বাংলা পাঠ্য পরীক্ষা।",
    "ภาษาไทยสำหรับการทดสอบ",
    "Tiếng Việt với dấu: à á ả ã ạ ề ể ễ ệ",
    "Türkçe karakterler: ğüşıöç İ",
]

SPECIAL_STRINGS = [
    "Specials inline: <|endoftext|> in the middle",
    "[CLS] bracket specials [SEP]",
    "</s> end of sentence <pad>",
    "<s> start",
    "email@example.com and user+tag@domain.co.uk",
    "C++23 vs C++20 vs C++17",
    "a=b==c!=d<=e>=f&&g||h",
]

# Code points that force byte-fallback tokens in byte-level vocabularies and
# exercise multi-token UTF-8 splits in streaming decode.
BYTE_FALLBACK_POOL = (
    [chr(c) for c in range(0x1F300, 0x1F6D0)]     # emoji blocks
    + [chr(c) for c in range(0x20000, 0x20050)]   # rare CJK ext-B
    + [chr(c) for c in range(0x2B740, 0x2B790)]   # rare CJK ext-D
    + ["∀", "∂", "∈", "ℝ", "∧", "∪", "≡", "⊂", "ℵ", "Ⅷ", "⇔", "─", "┬"]
    + ["ﬁ", "ﬂ", "ﬀ", "Æ", "ß", "Ø", "œ", "￥", "£", "€"]
    + ["\u0301", "\u0302", "\u0308"]              # combining marks
)

WORD_POOL = [
    "attention", "softmax", "rmsnorm", "rope", "kv", "cache", "page", "table",
    "scheduler", "preemption", "throughput", "latency", "fingerprint",
    "safetensors", "tokenizer", "vocabulary", "byte", "fallback", "stream",
    "decode", "encode", "chunk", "prefix", "batch", "tensor", "kernel",
]


def random_text(rng: random.Random) -> str:
    kind = rng.randrange(10)
    if kind == 0:
        return rng.choice(ENGLISH)
    if kind == 1:
        return rng.choice(MULTILINGUAL)
    if kind == 2:
        return rng.choice(SPECIAL_STRINGS)
    if kind == 3:
        # Byte-fallback / rare-codepoint stress.
        count = rng.randrange(1, 12)
        return "".join(rng.choice(BYTE_FALLBACK_POOL)
                       for _ in range(count))
    if kind == 4:
        # Combining-mark suffixes on Latin base letters.
        base = rng.choice(["a", "e", "i", "o", "u", "n", "s", "z"])
        marks = "".join(rng.choice(["\u0300", "\u0301", "\u0308", "\u0327"])
                        for _ in range(rng.randrange(1, 4)))
        return base + marks
    if kind == 5:
        # Random valid code points (excluding surrogates and noncharacters).
        count = rng.randrange(1, 40)
        out = []
        for _ in range(count):
            block = rng.choice([0x20, 0x2000, 0x4E00, 0x3040, 0x1F600, 0x10000])
            cp = block + rng.randrange(0, 0x50)
            if 0xD800 <= cp <= 0xDFFF or cp in (0xFFFE, 0xFFFF):
                continue
            out.append(chr(cp))
        return "".join(out)
    if kind == 6:
        # Whitespace edge cases.
        sep = rng.choice([" ", "  ", "\t", "\n", "\u00A0", "\u3000", " \n "])
        words = [rng.choice(WORD_POOL) for _ in range(rng.randrange(1, 10))]
        return rng.choice(["", " ", "\n"]) + sep.join(words) + rng.choice(["", " ", "\n"])
    if kind == 7:
        # Long single "word" (subword stress).
        return "".join(rng.choice(WORD_POOL) for _ in range(rng.randrange(4, 20)))
    if kind == 8:
        # Repeated token stress.
        word = rng.choice(WORD_POOL)
        return " ".join([word] * rng.randrange(2, 30))
    # Mixed everything.
    parts = [
        rng.choice(ENGLISH), rng.choice(MULTILINGUAL),
        "".join(rng.choice(BYTE_FALLBACK_POOL) for _ in range(3)),
        rng.choice(WORD_POOL),
    ]
    rng.shuffle(parts)
    return " ".join(parts)


def main() -> int:
    import tokenizers

    checkpoints = sorted(
        d.name for d in REFERENCE.iterdir() if (d / "tokenizer.json").exists()
    )
    if not checkpoints:
        print("no checkpoints found", file=sys.stderr)
        return 1

    total = 0
    for name in checkpoints:
        directory = REFERENCE / name
        tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
        # zlib.crc32 is stable across processes, unlike salted hash().
        rng = random.Random(SEED + zlib.crc32(name.encode()) % 100007)

        with (directory / "cases.jsonl").open("w", encoding="utf-8") as out:
            for _ in range(CASES_PER_CHECKPOINT):
                text = random_text(rng)
                ids_special = list(tok.encode(text, add_special_tokens=True).ids)
                ids_plain = list(tok.encode(text, add_special_tokens=False).ids)
                decoded = tok.decode(ids_plain, skip_special_tokens=False)
                record = {
                    "text": text,
                    "ids_special": ids_special,
                    "ids_plain": ids_plain,
                    "decoded_plain": decoded,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        total += CASES_PER_CHECKPOINT
        print(f"{name}: {CASES_PER_CHECKPOINT} cases")

    manifest = {
        "oracle": "huggingface tokenizers (python)",
        "oracle_version": tokenizers.__version__,
        "engine": "tokenizers crate 0.21.2 (rust, behind inferx-tokenizers-c)",
        "seed": SEED,
        "cases_per_checkpoint": CASES_PER_CHECKPOINT,
        "checkpoints": checkpoints,
        "total_cases": total,
    }
    (REFERENCE / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"total: {total} cases across {len(checkpoints)} checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
