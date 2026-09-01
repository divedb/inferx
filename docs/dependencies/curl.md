# Qualification report: curl

- Manifest entry: `curl` — approved, feature `core`, owner `artifacts-model-resolution`
- Pin: `68720b4837284335b2d63cb358f8f6ce65f5bc55` (signed release tag `curl-8_21_0`)
- License: curl license (`COPYING` at the pin)

## 1. Which InferX contract uses it?

The private production transport for Hugging Face model metadata and artifact downloads. No curl
type appears in an installed InferX header. Local and cache-hit resolution do not enter curl.

## 2. Required, deferred, experimental, or rejected?

Approved when `INFERX_ENABLE_HF_HUB=ON`, which is the top-level default so a server-style model ID
works without a separate download step. Embedders may disable the feature and retain local/cache
resolution.

## 3. Source and transitive dependencies

Pinned direct submodule at stable curl 8.21.0, promoted from the transport closure previously hidden
inside `divedb/tokenizer`. InferX builds only the static library, with HTTP(S), proxy support, and
OpenSSL 3.0 or newer. The curl executable, tests, examples, non-HTTP protocols, libpsl, SSH, zlib,
Brotli, Zstandard, and generated manuals are disabled. System-provider builds require CURL 8.21 or
newer. curl's Git source tree reports `8.21.0-DEV` during configuration because release archives are
version-stamped separately; provenance is the exact `curl-8_21_0` annotated release tag and commit.

## 4. Toolchain/C++23 compatibility

curl is C and builds cleanly as a `SYSTEM` dependency under the accepted GCC 13/Clang 18 host
toolchains. Its options are normal scoped variables and are restored after `add_subdirectory`; no
cache force or filesystem mutation is used.

## 5. Runtime behavior caveats

InferX initializes curl once, uses one easy handle per thread, resets request options between calls,
restricts initial and redirected schemes to HTTP(S), verifies TLS through OpenSSL, caps redirects,
and streams model files to unique staging paths. Metadata is bounded at 16 MiB. Resolver retries are
bounded and limited to transport failures, HTTP 429, and HTTP 5xx.

## 6. API stability and namespaces

Only the stable easy API is used. `curl_http_client.cc` is the sole translation unit including
`curl/curl.h`; `HttpClient` is an InferX-private hermetic test seam.

## 7. License, notice, and security

The curl license permits use and redistribution when its copyright/permission notice is retained;
InferX installs that notice as `third-party-notices/curl-COPYING` for submodule-provider packages.
As of the 2026-09-01 audit, curl's official
[vulnerability table](https://curl.se/docs/vulnerabilities.html) lists no published vulnerability
for 8.21.0. The pin is the current stable release; the inherited 8.22.0 development commit was
rejected because 8.22.0 remained pending.

## 8. Binary/build/startup cost

The static HTTP(S)-only archive adds native download support but no cost on local/cache hits beyond
the resolver itself. Exact size and cold-download throughput remain release evidence rather than a
correctness gate.

## 9. Upgrade and rollback

Advance the gitlink and manifest together to a signed stable tag, review the official curl security
table, build both source/system providers, run the hermetic resolver suite and one opt-in Hub smoke,
then regenerate the SBOM. Rollback disables `INFERX_ENABLE_HF_HUB` while retaining local/cache
resolution.

## 10. Disposition

Approved for the narrow private HTTP(S) transport above. It is not a general InferX networking
layer and server HTTP remains owned by the server module.
