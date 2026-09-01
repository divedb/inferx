# Model resolution

InferX accepts the same basic model spelling as vLLM: a model argument is either a local directory
or a case-preserving Hugging Face model ID such as `Qwen/Qwen2.5-0.5B-Instruct`. The M3 inspection
tool exposes the contract now; the server entry point will consume the same `ModelResolver` API.

```sh
inferx-model-inspect Qwen/Qwen2.5-0.5B-Instruct
inferx-model-inspect Qwen/Qwen2.5-0.5B-Instruct --revision REVISION
inferx-model-inspect ./models/tiny-llama --offline
inferx-model-inspect Qwen/Qwen2.5-0.5B-Instruct --download-dir /models/hf-cache
```

Resolution is deterministic and local-first:

1. use `MODEL` directly when it names an existing directory;
2. try `MODEL` below each colon-separated `INFERX_MODEL_DIR` entry;
3. look up the requested revision in the Hugging Face cache;
4. when it is absent and offline mode is not active, download the supported snapshot files from the
   Hub and reuse that cache on later starts.

Values beginning with `.`, `/`, `~`, a Windows drive prefix, or containing a backslash are explicit
paths. A missing explicit path fails locally and is never reinterpreted as a Hub ID. Repository IDs
are never lowercased because Hugging Face cache names preserve casing.

`--download-dir` has the same role as vLLM's option of that name. Without it, cache selection follows
Hugging Face precedence: `HF_HUB_CACHE`, then `$HF_HOME/hub`, then
`$XDG_CACHE_HOME/huggingface/hub`, then `~/.cache/huggingface/hub`. `--offline` and
`HF_HUB_OFFLINE=1` prohibit every HTTP call. `--revision` accepts a branch, tag, full commit SHA, or
safe slash-qualified ref. A full cached commit SHA starts without a network request.

Authentication precedence is the resolver option, `HF_TOKEN`, the deprecated
`HUGGING_FACE_HUB_TOKEN`, then `HF_TOKEN_PATH` or `$HF_HOME/token`. Tokens are sent only as an
authorization header and never appear in diagnostics. `HF_ENDPOINT` selects a compatible Hub
endpoint. libcurl handles HTTPS, redirects, standard proxy environment variables, and certificate
verification; InferX limits redirects to HTTP(S).

InferX downloads configuration/tokenizer JSON, safetensors, and the non-executable tokenizer assets
needed to load a model. It deliberately ignores pickle/`.bin`, checkpoint Python, Markdown, and
`original/` duplicate weights. It never executes repository code. The selected files follow the
same cache `refs`/`blobs`/`snapshots` layout used by `huggingface_hub`, so existing downloads are
reused. See the official [Hugging Face cache schema](https://huggingface.co/docs/hub/local-cache),
[download contract](https://huggingface.co/docs/huggingface_hub/guides/download), and
[environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

Hugging Face snapshots normally expose blob files through descendant symlinks. M3's rooted loader
correctly rejects descendant symlinks, so the resolver atomically creates an `inferx/snapshots/<sha>`
view containing only regular hard links (or copies when hard links are unavailable). The loader sees
that symlink-free immutable view; the standard cache remains interoperable with other tools.

The resolver can locate any Hub repository with the required safe artifacts, but the current model
parser still accepts only M3's dense Llama capability. Resolution succeeding does not imply that a
Qwen2 or another architecture is executable yet. `INFERX_ENABLE_HF_HUB=OFF` keeps local and cached
resolution but returns `Unimplemented` for a cache miss instead of linking an HTTP/TLS transport.

The positional model convention, `--revision`, and `--download-dir` intentionally track the
[vLLM serve interface](https://docs.vllm.ai/en/latest/cli/serve/). InferX does not claim behavioral
compatibility for vLLM features outside this model-source contract.
