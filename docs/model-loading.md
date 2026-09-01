# Dense Llama model loading

`ModelArtifactLoader::Inspect` performs deterministic discovery, content hashing, strict parsing,
external tensor catalog construction, and complete logical planning. Required fixed inputs are
`config.json` and `tokenizer.json`. An index is preferred over standalone weights only when there is
no ambiguity; if both exist, a manifest must explicitly select `model.safetensors.index.json`.
Every ordinary JSON artifact is size-checked before hashing or parsing, and every consumed file plus
the optional manifest is identity-checked again immediately before a successful inspection returns.

The artifact-foundation merge accepts `ArtifactLimits` directly at the loader boundary. These limits
are not yet part of M1's effective engine configuration or provenance record; that integration is a
remaining full-M3 task before `ValidatedModelPackage` readiness is published.

The initial schema accepts `model_type=llama` and, when declared, exactly
`LlamaForCausalLM`. It consumes vocabulary/hidden/intermediate sizes, layer and attention counts,
head dimension, maximum positions, RMS epsilon, RoPE theta, dropout provenance, embedding ties,
special IDs, and a recognized floating dtype hint. `num_key_value_heads`, `head_dim`, `rope_theta`,
`hidden_act`, biases, ties, and `pretraining_tp` use the defaults and restrictions in M3. Unknown
fields fail until classified. Known inert metadata such as `_name_or_path` and
`transformers_version` cannot override semantic values.

For each layer the planner requires the two normalization weights, Q/K/V/O projections, and
gate/up/down MLP projections, plus embedding, final norm, and LM head. Exact names and shapes are
used; no regex/prefix ignore rule exists. All weights must be uniformly F16, BF16, or F32. Missing,
duplicate, unexpected, mixed-dtype, rank, shape, and byte-size mismatches fail transactionally.

With tied embeddings, an absent LM head becomes an alias. If present, its tensor-content digest must
equal the embedding tensor digest. Plan items use identity transforms and replicated full logical
shards. `InspectedModelArtifacts` is intentionally not model readiness: tokenizer qualification and
model/tokenizer cross-checks must occur before a future `ValidatedModelPackage` is published.

The experimental static `inferx::artifacts` and `inferx::model` targets are installed with the public
headers. The installed-package consumer exercises both targets and their pinned simdjson/BLAKE3 link
closure without source-tree or build-tree paths.
