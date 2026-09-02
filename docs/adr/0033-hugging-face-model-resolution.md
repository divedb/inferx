# ADR 0033: Hugging Face model resolution and cache handoff

- Status: Accepted
- Date: 2026-09-01
- Owner: artifacts/model resolution

## Context

M3 initially required an explicitly selected local directory and excluded model-name resolution.
That made server usage diverge from vLLM and required an operator to run a separate downloader. The
rejected `divedb/tokenizer` dependency already contained useful MIT-licensed Hub authentication,
retry, cache-layout, and atomic-staging code, but its resolver downloaded tokenizer files only and
its larger tokenizer/build closure could not be accepted.

The standard Hugging Face snapshot cache contains descendant symlinks into content-addressed blobs,
while ADR 0034's loader correctly rejects every descendant symlink. Passing a standard snapshot
directly to the loader would silently violate one of those contracts.

## Decision

InferX owns a model resolver before the ADR 0034 trust boundary. `MODEL` is an existing local
directory, a name below configured local model directories, or a case-preserving Hugging Face repo
ID. Resolution checks local directories and the standard Hugging Face cache before making a network
request. `HF_HUB_OFFLINE` and the explicit offline option make cache misses fail without HTTP.

The resolver adapts the rejected dependency's MIT-licensed cache, token, retry, diagnostic, and
libcurl transport implementation with attribution. Repository metadata enumerates a safe model
artifact subset including JSON, tokenizer assets, and safetensors; executable code, pickle, and
duplicate `original/` weights are not downloaded. Native HTTPS uses a directly pinned stable curl
release rather than retaining the tokenizer dependency or invoking Python/a shell command.

Downloaded files use the standard Hugging Face `refs`/`blobs`/`snapshots` layout. Before loading,
InferX atomically materializes a second symlink-free view under `inferx/snapshots/<commit>` using
hard links or copies. Only this regular-file view crosses into `ModelLocator`; ADR 0034 remains
unchanged inside its boundary.

## Alternatives

- Requiring pre-downloads was rejected because it does not satisfy server-style model naming.
- Invoking `hf`, Python, or `curl` executables was rejected because it adds runtime/tooling coupling,
  error parsing, and process-control risk.
- Passing standard cache symlinks to `ModelLocator` was rejected because it weakens ADR 0034.
- Copying entire model repositories was rejected because it admits executable/pickle files and
  duplicate weight formats that InferX will never load.
- Keeping the inherited curl development pin was rejected in favor of the current signed stable
  release.

## Consequences

The CLI/server-facing convention can follow `COMMAND MODEL --revision ... --download-dir ...`.
Existing Hugging Face caches are reusable, cold starts automatically download supported files, and
offline deployments remain deterministic. The symlink-free view consumes directory entries and may
consume full duplicate storage on filesystems where hard links are unavailable. curl/OpenSSL become
an optional production closure; local/cache-only embedders can disable it.

Model resolution and architecture support remain separate: a repository may resolve successfully
and then fail the current dense-Llama capability check.

## Validation evidence

`ModelResolverTest` covers direct/local-directory priority, explicit-path typo handling, reference
cache interoperability, symlink-free materialization, offline misses with zero transport calls,
cold download followed by a cache hit, file allowlisting/path rejection, and case preservation.
Default and `INFERX_ENABLE_HF_HUB=OFF` builds, install consumers, sanitizer lanes, manifest drift,
and CLI tests are release gates.

## Supersession

A replacement may introduce another compatible Hub transport or immutable content store, but it
must preserve local-first resolution, offline zero-network behavior, safe artifact filtering, and
the ADR 0034 regular-file handoff.
