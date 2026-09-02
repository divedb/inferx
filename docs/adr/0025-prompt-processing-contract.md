# ADR 0025: prompt processing contract

- Status: Accepted
- Date: 2026-08-31
- Owner: input processing

## Context

Raw text, chat templates, and caller-owned token IDs have different special-token and validation
semantics. Conflating them creates double BOS/EOS insertion and lets text leak into scheduling.

## Decision

A request contains exactly one of raw UTF-8 text, text-only chat messages, or owned token IDs. Raw
text is encoded once with its explicit special-token policy. Chat selects only an explicit template
or a checkpoint-declared default, renders within a byte bound, then encodes once with special-token
insertion disabled. Token IDs bypass rendering and tokenization but not model, vocabulary, context,
deadline, or request validation.

There is no inferred template, family-name switch, string-concatenation fallback, double BOS, runtime
vocabulary mutation, or partial result. Limits are enforced before and after rendering and after
tokenization, including checked prompt-plus-output context arithmetic.

## Alternatives

- Guessing a template by model name was rejected because formatting errors produce valid-looking but
  incorrect generations.
- Encoding rendered chat with normal special-token insertion was rejected because templates own
  their emitted control tokens.
- Temporary M3 request/ID/channel types were rejected because M1 owns those contracts.

## Consequences

The implementation must consume M1's IDs, request, deadline, bounded channel, epoch, and lifecycle
types. Those interfaces are now present on the integrated branch; the artifact/model foundation
intentionally does not create temporary competing types while the tokenizer backend remains blocked.

## Validation evidence

The contract is currently documentation-only. Its acceptance tests activate after the ADR 0024
backend is available; until then no input target or readiness path is published.

## Supersession

New modalities or tool/reasoning objects require an explicit prompt-variant and template-capability
ADR; they cannot be accepted as arbitrary checkpoint objects.
