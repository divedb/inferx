# M4 operator contracts

This document is the implementation-facing summary of ADR 0026. Public requests use M2
`TensorView`/`MutableTensorView`; views do not own storage. CPU calls complete synchronously. CUDA
launch success means work was queued on the caller's stream, so the caller retains every buffer,
plan, metadata/workspace lease, and final `CompletionFence` until acknowledgement.

## Common policy

- Activations and dense outputs are contiguous row-major. Weights are `[out,in]`; linear and logits
  compute `A * B^T`.
- CPU reference operation tensors are FP32, except signed INT32 token IDs and positions. CUDA
  storage may be FP32, FP16, or BF16 only where its capability says so; accumulation is FP32.
- Unsupported dtype/layout/backend envelopes return `Unimplemented`. Malformed shapes, metadata,
  ranges, devices, or overlaps return `InvalidArgument`. Checked size/FLOP/workspace exhaustion
  returns `ResourceExhausted`.
- Zero tokens are fully validated no-ops. Vocabulary, hidden, intermediate, head count, head
  dimension, and KV context dimensions are positive where used.
- Validation occurs before mutation. Partial byte overlap is never legal. CUDA calls never allocate,
  synchronize, create a stream, or perform live heuristic search.

## Operation matrix

| Operation | Inputs and output | Formula/phase | Legal alias | Workspace |
|---|---|---|---|---:|
| embedding | IDs `[T]` INT32, weight `[V,H]`, output `[T,H]` | validate every ID, then copy exact row | none | 0 |
| GEMM | A `[M,K]`, B `[N,K]`, optional C `[M,N]`, D `[M,N]` | `D=alpha*A*B^T+beta*C`; finite scales | none | backend plan; bounded by key |
| RMSNorm | x `[T,H]`, weight `[H]`, y `[T,H]` | FP32 `x/sqrt(mean(x*x)+eps)*weight` | none | 0 |
| RoPE | Q `[T,Nq,D]`, K `[T,Nkv,D]`, positions `[T]` | half-split `(i,i+D/2)`, positive theta, bounded positions | each output may exactly alias its matching input | 0 |
| SiLU | x and y equal positive-rank shape | `x/(1+exp(-x))` | exact input/output | 0 |
| multiply/SwiGLU | left/right/output equal shape | `left*right` or `silu(left)*right` | exact output/one input if disjoint from the other | 0 |
| residual | left/right/output equal shape | `left+right` | exact output/left or output/right | 0 |
| attention | Q `[Tq,Nq,D]`, new K/V `[Tnew,Nkv,D]`, K/V cache `[B,S,Nkv,D]`, output `[Tq,Nq,D]` | causal prefill/decode, contiguous rotated-K append, stable GQA softmax | none; cache writes only validated slots | CPU: `S` FP32 scores; owned CUDA fallback: 0 |
| logits | hidden `[T,H]`, LM head `[V,H]`, output `[T,V]` | named `hidden*weight^T`, FP32 baseline output | none | cuBLASLt plan; bounded by key |

CUDA embedding and RoPE requests retain exact immutable host mirrors of token IDs and positions.
The mirrors let launch reject semantic range errors before queueing work; the owned kernels also
scan the device vectors before writing as a defense against a broken mirror/transfer contract.

## Attention metadata

`query_indptr` and `new_kv_indptr` each have `B+1` nondecreasing signed INT32 entries beginning at
zero and ending at their flattened token count. `kv_lengths_before` has `B` nonnegative entries;
`query_positions` has one entry per query. For the initial contract each sequence has equal query and
new-KV counts and contiguous positions starting at its previous KV length. Decode admits zero or one
query per sequence. Visible keys for absolute position `p` are exactly `[0,p]`.

The CUDA fallback envelope is `1<=B<=8`, `1<=Nkv<=Nq<=64`, `Nq%Nkv==0`, even `2<=D<=256`,
`S<=4096`, prefill total queries `<=512`, and at most one decode query per sequence. The K cache
stores K after RoPE; cached K is never rotated again.

## Numeric comparison maxima

| Storage/operation | `atol` | `rtol` |
|---|---:|---:|
| FP32 copy/elementwise/residual/RoPE | `2e-6` | `2e-5` |
| FP32 RMSNorm/SiLU | `2e-5` | `2e-4` |
| FP32 GEMM/logits | `2e-4` | `2e-4` |
| FP32 attention | `3e-4` | `3e-4` |
| FP16 storage, FP32 accumulation | `2e-2` | `2e-2` |
| BF16 storage, FP32 accumulation | `8e-2` | `5e-2` |

Embedding copies must be bit-exact. Numeric tests additionally compare finite classification,
absolute/relative worst cases, and operation-specific diagnostics; changing a maximum requires an
independent error study and a superseding ADR.
