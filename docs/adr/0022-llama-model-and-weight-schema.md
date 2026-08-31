# ADR 0022: Llama model and weight schema

- Status: Accepted
- Date: 2026-08-31
- Owner: model

## Context

Model repositories contain generic metadata, architecture-affecting options, converted buffers, and
training state. Guessing which unknown tensors or fields are harmless risks constructing a plausible
but semantically wrong model.

## Decision

The first architecture is dense, bias-free Hugging Face `LlamaForCausalLM` with `model_type=llama`.
Config fields are divided into consumed, inert, and recognized-but-unsupported sets. Unknown fields
fail until classified. Normalized defaults and derived head/divisibility/token checks form model
schema version 1.

The parameter catalog names every embedding, per-layer normalization/attention/MLP weight, final
normalization, and LM head with its exact rank and shape. Source weights are one uniform F16, BF16,
or F32 family. Every expected parameter is assigned once and every external tensor is consumed once.
There are no prefix ignore rules.

When embeddings are tied, an omitted `lm_head.weight` becomes an alias. A present LM head is accepted
only when its content digest equals the embedding content digest.

## Alternatives

- Repository-name dispatch and regex/prefix ignores were rejected as non-semantic and overbroad.
- Accepting any Transformers architecture was rejected until each parameter and feature has an
  execution contract.
- Shape-only tied-weight detection was rejected because equal dimensions do not imply equal values.

## Consequences

Biases, rotary buffers, optimizer state, quantization, mixtures of experts, sliding windows, and RoPE
scaling fail explicitly. M3 plans only an identity transform and replicated full logical shard.

## Validation evidence

`ModelArtifactLoaderTest.BuildsCompleteTinyLlamaWeightPlan` proves the baseline schema reconciles all
12 parameters in a one-layer fixture. Missing, unexpected, dtype, mixed-family, shape, and tie-digest
cases are owned by the expanding M3 corpus.

## Supersession

New architectures or accepted buffers require a new schema version and ADR; existing unknowns do not
become inert implicitly.
