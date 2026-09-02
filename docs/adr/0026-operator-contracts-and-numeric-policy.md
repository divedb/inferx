# ADR 0026: operator contracts and numeric policy

- Status: Accepted
- Date: 2026-09-01
- Owner: operator runtime

## Context

M5 needs one mathematical vocabulary shared by the independent CPU oracle and every prepared
backend. Inferred transposes, implicit promotion, loose aliasing, or backend-specific attention
semantics would make a plausible result impossible to diagnose.

## Decision

M4 uses M2 tensor views with row-major activations, `[out,in]` dense weights, Q/K/V
`[tokens,heads,head_dim]`, and separate contiguous K/V
`[batch,max_context,kv_heads,head_dim]`. Dense linear and logits compute `A * B^T`. Llama RoPE uses
half-split pairs. Attention is causal, appends already-rotated K, maps GQA heads by contiguous query
groups, and uses stable max-subtracted softmax.

The CPU oracle accepts FP32 tensors and performs FP32 accumulation in a fixed traversal order. CUDA
FP16/BF16 reductions and GEMMs accumulate in FP32; TF32 and fast math are not baseline modes. Every
request validates rank, shape, dtype, contiguous layout, device, metadata, and allocation-identity
byte intervals before mutation. Only the exact aliases documented in
[`operator-contracts.md`](../operator-contracts.md) are accepted. Zero tokens are validated no-ops;
semantic hidden, vocabulary, and head dimensions remain positive.

## Alternatives

- Reusing a vendor implementation as the reference was rejected because it would not be an
  independent oracle.
- Adjacent-pair RoPE and transposed checkpoint weights were rejected because they do not match the
  supported Llama artifact contract.
- Implicit dtype promotion and TF32 were rejected because they hide numeric policy from the key.

## Consequences

Optimized backends may use different schedules but must advertise the same semantic contract and
meet M4's fixed tolerances. New layouts, math modes, fusions, masks, or alias modes require distinct
capabilities and correctness evidence. The reference remains intentionally bounded and slow.

## Validation evidence

`M4ReferenceTest.*` covers transactional embedding validation, `[out,in]` GEMM, RMSNorm/SwiGLU,
half-split RoPE, causal GQA attention, and KV append. The GCC C++23 CPU build, eight M4 tests, and all
75 public-header self-containment probes passed on 2026-09-01. Supported-GPU numeric and sanitizer
evidence remains owned by the CUDA 13 CI lane and is not inferred from these CPU results.

## Supersession

A later ADR may add a math mode, layout, or operator semantic only with a new key value, independent
oracle evidence, and an explicit compatibility statement for M5.
