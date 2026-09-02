# simdjson qualification

- Revision: `0a851a64cd984e9e1a6cab93b6e773aa3f4dc30d` (`v4.6.5`)
- License: Apache-2.0
- Owner: configuration and artifacts
- Status: approved

InferX uses simdjson only behind schema-specific readers. No simdjson type is present in a public
header and no view survives its parser. The source build is static and offline, with installation,
internal threads, deprecated APIs, and developer targets disabled. Every reader performs an explicit
file-size check, parser-depth allocation, duplicate-key walk, exact integer conversion, and immediate
copy into owned values.

The pin builds in C++23 host mode with GCC and Clang. Upgrade qualification must rerun configuration,
malformed JSON, duplicate-key, UTF-8, large-integer, maximum-depth, safetensors, shard-index,
manifest, and Llama configuration corpora under ASan/UBSan.
