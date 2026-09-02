# Qualification report: minja

- Manifest entry: `minja` — feature `tokenization` (M3), owner `tokenization`
- Pin: `021c2293c187789ef13d56c6cfd89c9b134fd80f` (submodule `third_party/minja`)
- License: Apache-2.0 (`LICENSE` at the pin)
- Decision: **Approved (ADR 0024, 2026-09-02)**

## Use and boundary

Google's header-only minimal Jinja engine: the *vetted template
implementation* m3.md section 8.3 requires for chat-template rendering
("use a vetted template implementation or a deliberately supported subset;
silently approximating Jinja templates is forbidden"). Included only by
`src/chat/chat_template.cc` inside the tokenizer vendor package; no other
translation unit reaches it and no InferX header exposes it.

## Why this pin

The exact revision divedb/tokenizer qualified its chat rendering against;
keeping the pair together preserves the adapted code's tested behavior.
Chat-template failures are typed errors (FailedPrecondition for a missing
template, InvalidArgument for invalid message shapes) — never a fallback
template.

## Security

Templates come from the checkpoint (`chat_template.jinja` /
`tokenizer_config.json`) inside the rooted session; rendering is bounded
(`max_rendered_bytes` checked after render, before encode) and result errors
surface as statuses.
