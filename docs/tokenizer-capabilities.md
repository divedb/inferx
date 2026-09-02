# Tokenizer capabilities (M3)

The qualified backend (ADR 0024) is an owned local-only adaptation of
divedb/tokenizer's C++ over an owned error-returning Rust FFI shim on the
official Hugging Face `tokenizers` engine. This file names every accepted
component/type and the tested engine versions; encountering an unlisted
construct returns `Unimplemented` with its tokenizer JSON path.

## Engine identity

| Component | Version | Pinned by |
|---|---|---|
| Rust `tokenizers` engine | 0.21.2 | `third_party/tokenizer/rust/Cargo.lock` + vendored closure |
| Python differential oracle | 0.21.1 (`tokenizers==0.21.1`) | `tools/fixtures/tokenizer-requirements.lock` |
| Owned FFI shim | `inferx-tokenizers-c` 0.1.0 | `third_party/tokenizer/rust` |
| Adapted C++ layer | divedb/tokenizer @ `f109b7ae` (MIT) | `third_party/tokenizer` |

## Accepted serialized components (conformance matrix)

Component types exercised exactly by the committed 10,000-case corpus
(`tests/fixtures/model/tokenizer_reference/`):

| Slot | Accepted types | Evidence checkpoint |
|---|---|---|
| `model` | `BPE`, `WordPiece`, `Unigram` | qwen2.5 / bert-base-uncased / t5-small |
| `normalizer` | `NFC`, `Sequence` (incl. `Prepend`/`Replace`), `BertNormalizer`, `Precompiled` | qwen2.5 / tinyllama / bert-base / t5 |
| `pre_tokenizer` | `Sequence`, `ByteLevel`, `BertPreTokenizer`, (absent) | qwen2.5 / bert-base / tinyllama |
| `post_processor` | `ByteLevel`, `TemplateProcessing` | qwen2.5 / tinyllama, bert-base, t5 |
| `decoder` | `ByteLevel`, `Sequence` (incl. `ByteFallback`, `Replace`, `Fuse`, `Strip`), `WordPiece`, `Metaspace` | qwen2.5 / tinyllama / bert-base / t5 |
| `added_tokens` | added-token array with id/content/special | all |

A `tokenizer.json` whose `model` object carries no `"type"` tag (as the
Hub's `gpt2` file historically did) is rejected as `DataLoss` by the engine;
the corpus checkpoints all carry well-formed tagged pipelines.

## Product capability (M3 surface)

- local `tokenizer.json` is required; `tokenizer.model`-only (SentencePiece)
  checkpoints are unsupported;
- encoding accepts valid UTF-8 only (`InvalidArgument` otherwise); decode
  rejects negative ids (`InvalidArgument`);
- streaming decode is the upstream `step_decode_stream` state machine with
  cleanup disabled (`clean_up_tokenization_spaces=false` serving default);
  the concatenation of all chunks plus `Finish()` is byte-exactly one-shot
  decode; every nonempty chunk is valid UTF-8; prior bytes never change;
- runtime vocabulary additions, stochastic tokenization, text pairs,
  training, remote loading, and `trust_remote_code` are unsupported;
- chat rendering uses the checkpoint-declared template (minja) or an
  explicitly named one; there is no fallback template;
- special-token resolution merges `tokenizer.json` added_tokens,
  `special_tokens_map.json`, `tokenizer_config.json`, explicit overrides,
  and `config.json` ids, in increasing precedence, with the engine's
  vocabulary authoritative for ids.

## Metadata record

`inferx::tokenization::TokenizerMetadata` is derived once from the same
engine resolution encode uses; its canonical form
(`CanonicalMetadataRecord`, schema v1) enters the model fingerprint as the
`tokenizer_capability` record.
