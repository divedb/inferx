# InferX production inference roadmap

Status: architecture and implementation roadmap. This document is the source of truth until
individual design documents and ADRs supersede a section. Every milestone below must leave the
repository buildable and independently verifiable.

## 1. Historical repository baseline and working assumptions

This section records the repository baseline from 2026-08-30 that drove the roadmap; it is not a
current inventory. M0, M1, and M2 are now implemented. M3's artifact/model foundation is implemented
and independently usable, while full M3 remains blocked on tokenizer qualification. Current status
and evidence live in each milestone document.

- There is no build system, C++ source, test, benchmark, CI configuration, or public API yet.
- `.clang-format` selects Google style but still says `Standard: c++20`; M0 must change it to
  C++23 and make formatting reproducible.
- `.clang-tidy` contains a useful initial warning set, but it is not yet run by any target or CI.
- `.gitmodules` declares `abseil-cpp`, `beast`, `cutlass`, `flashinfer`, `folly`, `hpc-ops`, and
  `tokenizer`. All seven working directories are uninitialized. Their pinned commits, licenses,
  transitive dependencies, C++ APIs, and CUDA compatibility must be qualified before any is made
  a required dependency.
- `docs/plan.md` is currently untracked. M0 must decide whether documentation and dependency lock
  data are committed together.
- No `AGENTS.md` or other repository-local engineering instructions exist.

Initial product assumptions, to be confirmed by ADRs in M0, are:

| Area | Initial scope | Expansion point |
|---|---|---|
| Platform | Linux x86-64, NVIDIA CUDA GPUs | Other CPUs/accelerators only after the CUDA path is stable |
| Language | C++23 host code; CUDA C++ only below the platform boundary | No Python in the runtime; Python is allowed in reference tests and conversion tools |
| Models | Dense decoder-only Llama architecture, FP16/BF16 | Qwen dense, quantization, then MoE |
| Artifacts | Hugging Face `config.json`, `tokenizer.json`, safetensors and shard index | Additional safe formats through loader plugins; never load pickle in the server |
| Deployment | Embedded C++ engine, then one process/one GPU server | Multi-GPU, multi-node, and disaggregated serving later |
| API | Token-ID C++ API first, text API second, OpenAI-compatible HTTP third | Internal gRPC control plane for distributed operation |
| Correctness | Deterministic greedy generation compared with an independent reference | Statistical sampling and quantized tolerances later |

The first useful release is not “all features.” It is a reliable vertical slice:

```text
safetensors + tokenizer.json
        -> load one Llama model
        -> tokenize one prompt
        -> prefill + paged KV decode on one NVIDIA GPU
        -> greedy/sample tokens
        -> stream text through an embeddable C++ API and HTTP endpoint
```

Everything after that must preserve this path.

### Non-goals for the initial release

- A training/autograd framework, general tensor library, graph compiler, or PyTorch replacement.
- Python model authoring or a TokenSpeed-style static placement compiler.
- Arbitrary Hugging Face architectures, beam search, multimodality, adapters, or CPU inference.
- Writing GEMM, HTTP parsing, RPC, tokenization, or collective communication from scratch.
- Transparent recovery of an individual failed rank. Early distributed releases fail and drain the
  affected replica; elastic recovery is a later cluster feature.
- A stable ABI before the request, tensor, model-artifact, and backend contracts have survived the
  single-GPU milestones.

## 2. Lessons from existing systems: adopt selectively

The following conclusions come from primary project documentation and source, not from treating any
one system as a template.

| System | Applicable ideas | What not to copy now |
|---|---|---|
| vLLM | Separate API, engine-core, and GPU-worker responsibilities; make scheduler output explicit; iteration-level continuous batching; paged KV; prefix caching; worker-side execution; connector boundary for remote KV | Python multiprocessing/ZMQ topology, broad hardware matrix, legacy compatibility layers, and a feature-dense scheduler before the core invariants exist |
| SGLang | Prefix match is a scheduling input; a radix/prefix index pins immutable KV pages; chunked prefill participates in allocation and admission; cache namespace includes more than token IDs; distinguish prefill- and decode-oriented policies | Its language/frontend, Python object model, token-granular allocator variants, and structured-program feature surface in the first release |
| TensorRT-LLM | Executor loop split into scheduling, resource preparation, model execution, sampling, and output handling; capacity scheduling separated from microbatch choice; explicit async request API; preallocated KV pools; KV exchange separated from KV ownership and transport | TensorRT engine construction as the only backend, its large compatibility surface, and adopting implementation details tied to its Python runtime |
| TokenSpeed | A typed request/KV finite-state machine; single-writer C++ control plane; kernels behind a registry; local-SPMD/placement annotations as a future way to reduce handwritten parallel code | Its current Python execution plane and static compiler. TokenSpeed is a preview and these ideas must be validated independently |

Concrete consequences for InferX:

1. The scheduler emits an immutable `StepPlan`; it does not launch CUDA or call a model.
2. KV capacity is reserved transactionally while planning, and ownership changes only on explicit
   state transitions or completion fences.
3. The execution backend consumes tensor/token/KV metadata without knowing HTTP or user-facing
   request objects.
4. Prefix identity is content- and namespace-based; request identity is lifecycle-based; neither is
   a device pointer.
5. Prefill and decode share a local engine initially, but `ExecutionTarget` and `KvTransport` leave a
   clean path to disaggregation.
6. Parallel placement is explicit data in `ParallelPlan`; model definitions describe semantics and
   do not contain rank-specific `if` chains.

Ideas intentionally deferred include vLLM's breadth of accelerators, SGLang's structured execution
frontend, TensorRT-LLM's build/compiler stack, and TokenSpeed's static local-SPMD compiler. They add
substantial surface area without reducing risk on the first end-to-end path.

Primary references are collected in section 21.

## 3. Architectural invariants

These rules are design constraints, not suggestions.

1. **Policy is separate from mechanism.** Scheduling policy sees immutable resource snapshots and
   produces plans. Allocators, executors, and transports perform the work.
2. **The control plane is hardware independent.** `base`, `api`, `engine`, `scheduler`, model
   metadata, sampling policy, and KV identity cannot include CUDA/NCCL headers.
3. **Model semantics are separate from operators, and operators from kernels.** A Llama layer names
   RMSNorm/attention/MLP operations; backend selection chooses implementations.
4. **No raw address is an identity or ownership token.** Device addresses live only in non-owning
   views scoped by owning buffers, leases, and completion fences.
5. **One mutable owner per engine replica.** The coordinator event loop is the sole writer of
   request state, block tables, scheduler queues, and prefix-index metadata.
6. **GPU work is asynchronous by default.** A successful submission means work was queued. Memory
   and request state cannot be reused until its `CompletionFence` succeeds.
7. **Hot-path allocation is bounded.** Model weights, KV pages, metadata buffers, workspaces, events,
   and common graph buffers are pooled before serving; an iteration must not depend on `cudaMalloc`.
8. **Admission is resource-aware.** Accepted work has bounded queue, host-memory, KV, sequence,
   token, and deadline costs. Backpressure is explicit.
9. **Errors are typed and scoped.** Invalid input fails one request; corrupt model artifacts fail
   startup; CUDA context loss poisons and drains a replica; distributed rank failure fails its group.
10. **Correctness precedes optimization.** Every optimized backend is compared with an independent
    reference and can be disabled at runtime.
11. **No unbounded cardinality in telemetry.** Request IDs, prompts, token IDs, and tenant IDs never
    appear as metric labels.
12. **Every cross-process format is versioned.** Request envelopes, `StepPlan`, KV descriptors,
    placement manifests, and worker capabilities carry schema and model-layout versions.

## 4. Target source layout and dependency direction

Use one public include tree and small libraries so dependency rules can be enforced by CMake and
include-what-you-use.

| Subsystem | Owns | Must not own |
|---|---|---|
| `api` | Embeddable request handles and ordered response event contract | Scheduling policy, CUDA, HTTP |
| `config` | Parsing stages, validation, immutable effective configuration | Live mutable engine state |
| `artifacts` | Safe file/manifest reading, model fingerprint, weight/shard plan | Model execution or device scheduling |
| `tokenization` | Text/token conversion, special tokens, incremental decode | Request admission, GPU work, HTTP sockets |
| `engine` | Request registry/FSM, event loop, completion/output coordination | Kernel selection or protocol parsing |
| `scheduler` | Admission policy, budgets, candidate choice, immutable step plans | CUDA launch, transport I/O, owning requests |
| `kv` | Page state/leases, sequence block tables, prefix identity, reservation/transfer semantics | Attention math or user request protocol |
| `runtime` | Model instances, workers, plan submission, tickets, workspace lifecycle | HTTP/JSON or scheduling policy |
| `model` | Architecture validation, semantic modules, parameter mapping | Rank-specific communication calls or kernels |
| `ops` | Hardware-neutral operation/layout/workspace contracts | Model names, scheduling, or server values |
| `platform/cuda` | CUDA resources, memory, streams, backend dispatch, health | Request policy or HTTP/tokenizer behavior |
| `sampling` | Parameter semantics, RNG, logits processing, stop decisions | Model forward, sockets, KV allocation |
| `distributed` | Topology/placement, process groups, worker protocol, KV transport | Model semantics or public API compatibility |
| `server` | HTTP/SSE, limits, auth hooks, protocol mapping | Engine state mutation or GPU ownership |
| `telemetry` | Metric/trace/log contracts and exporters | Business policy or unbounded request data |

```text
CMakeLists.txt
CMakePresets.json
cmake/
docs/
  architecture/
  adr/
include/inferx/
  base/             status, result, IDs, time, checked arithmetic
  api/              stable C++ request/response facade
  config/           validated immutable configuration
  tensor/           dtype, shape, device-neutral tensor/buffer views
  artifacts/        manifests, safetensors, HF config, weight plans
  tokenization/     tokenizer interface and text streaming
  model/            model specs, registry, Llama semantic modules
  ops/              hardware-neutral operation contracts
  kv/               block tables, allocator, prefix identity, transfer contracts
  scheduler/        admission, policies, resource snapshots, step plans
  engine/           request FSM, event loop, output assembly, shutdown
  runtime/          executor/worker abstractions, memory planner
  sampling/         validation, processors, sampler contracts
  distributed/      topology, placement, process groups, worker protocol
  server/           protocol-neutral server facade
  telemetry/        metrics/tracing/logging contracts
src/                 implementations matching include/inferx/
platform/cuda/       the only production code that includes CUDA/cuBLASLt/NCCL
kernels/cuda/        custom CUDA kernels and generated kernel registry
apps/
  inferx_cli/
  inferx_server/
  inferx_worker/
tests/
  unit/
  integration/
  correctness/
  distributed/
  failure/
  stress/
benchmarks/
  micro/
  scheduler/
  model/
  serving/
tools/               artifact inspection/conversion and benchmark drivers
third_party/          pinned sources only; no locally modified hidden forks
```

Allowed dependency direction:

```text
apps/server -> api -> engine -> scheduler/runtime -> model/ops -> tensor/base
                         |             |                |
                         +-----------> kv <-------------+
runtime -> platform interface <- platform/cuda -> CUDA libraries and kernels
distributed -> runtime/kv contracts; networking never depends on platform/cuda
telemetry contracts may be used everywhere; exporters may not be used by core libraries
```

`base`, `tensor`, `scheduler`, and the logical portion of `kv` must have CPU-only build/test targets.
CUDA support is an optional CMake feature for developer machines; production presets require it.

## 5. Core contracts and coding rules

### 5.1 Language and style

- Compile host targets with `cxx_std_23`; isolate CUDA translation units if the selected NVCC cannot
  parse a required C++23 facility.
- Follow the Google C++ Style Guide and C++ Core Guidelines. `.clang-format`, `.clang-tidy`, IWYU,
  compiler warnings, and sanitizers are merge gates, not cleanup tasks.
- Prefer value types, `std::unique_ptr`, `std::span`, `std::string_view`, `std::chrono`, `std::jthread`,
  and `std::stop_token`. Use `std::shared_ptr` only when ownership is truly shared and documented.
- Raw pointers and references are non-owning and cannot outlive the call unless an interface says so.
- RAII-wrap CUDA streams/events/graphs, device and pinned buffers, cuBLASLt handles, NCCL
  communicators, file mappings, sockets, registrations, and trace spans.
- No exceptions cross public/runtime/plugin boundaries. A dependency that throws is caught in its
  adapter. Destructors never report recoverable failures and never synchronize unexpectedly.
- Avoid global mutable state. Registries are constructed at startup, frozen before serving, and
  passed explicitly.
- All size/offset calculations use checked arithmetic. Tensor dimensions and file offsets are
  validated before multiplication or allocation.

### 5.2 Status, IDs, and time

Use Abseil's `absl::Status` and `absl::StatusOr<T>` rather than inventing parallel error types.
Adapters translate `cudaError_t`, `cublasStatus_t`, `ncclResult_t`, filesystem, tokenizer, HTTP, and
RPC failures into canonical codes plus structured payloads such as component, device, rank, request
ID, and retryability. User-visible messages are sanitized; detailed causes go to logs/traces.

| Failure class | Scope/action | Retry contract |
|---|---|---|
| Invalid request/config value | Reject before resource admission; no engine-state change | Client may correct and resubmit |
| Deadline/cancellation | Coordinator transitions one request and releases after fences | Terminal; client chooses a new request ID |
| Queue/KV capacity | Reject at admission or keep in a bounded wait state according to policy | `RESOURCE_EXHAUSTED` may be retried with backoff |
| Corrupt/unsupported artifact | Fail model startup/readiness; tear down partial load by RAII | Retry only with corrected artifact/config |
| Per-step backend error with intact context | Fail affected requests, invalidate writes, collect diagnostics | Retry only if backend marks it safe |
| CUDA illegal access/device loss | Poison and drain the entire replica; supervisor restarts process | Requests return retryable `UNAVAILABLE` |
| Collective/rank divergence or timeout | Abort/drain the full parallel replica; no rank continues alone | Retry on another healthy replica |
| KV transfer timeout/corruption | Invalidate destination; recompute if deadline/capacity allow, else fail request | Internal bounded retry; never reuse uncertain pages |
| Telemetry/exporter failure | Drop/sample telemetry with counters; inference remains healthy | Exporter retries independently |

Define strong, non-interchangeable ID value types for at least `RequestId`, `SequenceId`, `ModelId`,
`ModelRevision`, `ReplicaId`, `WorkerId`, `Rank`, `DeviceId`, `BlockId`, `BlockGeneration`, and
`StepId`. Timestamps use `std::chrono::steady_clock` internally; wall time is telemetry only.

### 5.3 Tensor and buffer model

InferX needs inference views, not a general tensor framework.

```cpp
enum class DeviceKind { kHost, kCuda };
enum class MemoryKind { kHost, kPinnedHost, kDevice, kManaged };

struct TensorView {
  BufferView buffer;       // Non-owning, bounds-checked byte range.
  DType dtype;
  Shape shape;
  Strides strides;
};
```

`Buffer` uniquely owns an allocation through a device-neutral deleter. `BufferView` carries device,
offset, size, and alignment, but no ownership. Tensors are immutable by default; mutating operations
receive `MutableTensorView`. Shapes have a small-rank optimized representation and checked
`NumElements()`/`Bytes()`. Layout, quantization scales, and shard metadata are separate descriptors,
not hidden in `DType`.

### 5.4 Configuration

Configuration flows through three stages:

```text
CLI/environment/config file -> ParsedConfig -> Validate(capabilities) -> immutable EngineConfig
```

Unknown fields, inconsistent limits, impossible parallel topology, unsupported dtype/SM pairs, and
unsafe memory budgets are startup errors. Each effective config is redacted, fingerprinted, logged,
and exposed in diagnostics. Runtime tuning values may change only through an explicit atomic
`RuntimePolicySnapshot`; model layout and memory-pool geometry are immutable.

## 6. Process, thread, ownership, and shutdown model

### 6.1 Initial process topology

The single-GPU server starts as one process with separated components. Multi-GPU later uses one
worker process per GPU so a CUDA-context failure does not corrupt the API/coordinator process.

```text
HTTP I/O threads --bounded MPSC--> EngineCoordinator (one event-loop thread)
TokenizerPool ----bounded MPSC--/       |
                                       +--> DeviceWorker / ExecutionBackend
GPU completion queue <-----------------/        |
ResponseDispatcher <---- immutable events       +--> CUDA streams/events
```

- HTTP I/O threads own sockets and protocol parsing only.
- `TokenizerPool` owns tokenizer instances or immutable shared tokenizer state according to the
  selected library's thread-safety contract.
- `EngineCoordinator` is the only mutator of `RequestContext`, scheduler queues, KV block tables,
  prefix index, and accounting. Its hot loop does not take a contended mutex.
- `DeviceWorker` owns `DeviceContext`, model weights, streams, event/workspace pools, graph instances,
  and backend handles for one device. It accepts immutable plans through a bounded SPSC channel.
- Completion and response channels carry values/IDs, never pointers to scheduler containers.

The design permits the coordinator and worker to share a process initially. The channel interfaces
must not assume that; M14 replaces them with versioned IPC/RPC messages where necessary.

### 6.2 Ownership and lifetime table

| Object | Owner | Lifetime and access rule |
|---|---|---|
| `Engine` | Application | Starts components; `Shutdown()` is idempotent and joins all owned threads |
| `ModelInstance` | `DeviceWorker` | Created at startup; immutable weights live until all execution tickets finish |
| `RequestContext` | `RequestRegistry` inside coordinator | Stable allocation from admission through terminal response acknowledgement |
| prompt/output tokens | `RequestContext` | Mutated only by coordinator; plans contain ranges/snapshots, not vector iterators |
| physical KV pool | `KvCacheManager` | Fixed geometry per model/rank; destroyed after workers synchronize during shutdown |
| KV page reference | `KvLease`/prefix index | Coordinator-thread refcount; page reusable only at refcount zero and after write fence |
| execution metadata | `ExecutionTicket` | Pooled buffers retained until its completion fence; tagged with step and request epochs |
| model workspace | `WorkspacePool` | Lease per in-flight step/graph instance; not returned before completion |
| stream/event/graph | `DeviceContext` | RAII; graph captures retain stable addresses until graph cache teardown |
| response subscription | API/server layer | Cancellation posts an event; it never directly destroys a request or KV state |

`BlockId` is always paired with `BlockGeneration`. Reusing a physical slot increments the generation,
so stale plans and delayed transfer completions are rejected rather than corrupting a new request.

### 6.3 Synchronization rules

- Coordinator-owned state uses no atomics. Cross-thread cancellation/deadline events enter its queue.
- Bounded queues define backpressure. Each has a documented producer/consumer count and shutdown
  behavior; full admission queues return `RESOURCE_EXHAUSTED` rather than grow indefinitely.
- GPU dependencies use streams and events. Host synchronization is restricted to startup, controlled
  diagnostics, shutdown, and a clearly named test hook.
- Metadata H2D buffers are pinned and double/triple buffered. A slot cannot be overwritten until its
  event completes.
- Each distributed `StepPlan` includes a monotonically increasing collective sequence. All ranks
  execute the same ordered collective list, including zero-token participation when required.
- If locks are later necessary, the order is registry -> transport session -> exporter; device
  callbacks never acquire registry locks. Add a TSan test for every new shared mutable path.

### 6.4 Graceful shutdown

Shutdown has explicit phases: stop admission; cancel or drain by configured deadline; stop planning;
wait for in-flight GPU/transport fences; emit terminal responses; flush telemetry; destroy NCCL and
CUDA resources; join threads. Deadline expiry marks unfinished requests `UNAVAILABLE`, abandons
network delivery, and still waits for memory-safe device teardown. Tests must exercise shutdown at
every request state.

## 7. Request and response lifecycle

### 7.1 Public request values

`GenerateRequest` is an immutable value after validation:

```cpp
struct GenerateRequest {
  RequestId id;
  ModelId model;
  std::variant<std::string, std::vector<TokenId>> input;
  SamplingParams sampling;
  OutputOptions output;
  Priority priority;
  std::optional<TimePoint> deadline;
  TenantScope tenant;
};
```

Validation caps bytes, prompt tokens, output tokens, stop sequences, `logprobs`, and sampling values;
rejects duplicate IDs; and verifies the requested context fits the model. Server-generated globally
unique IDs are the default. User IDs are metadata and cannot key internal KV transfer.

### 7.2 Finite-state machine

Use a closed transition table, not independent booleans:

```text
Received -> Tokenizing -> Queued -> Reserving -> PrefillReady -> Prefilling
                                      ^              |              |
                                      |              +<-------------+ (next chunk)
                                      |                             v
                                  Preempted <---- DecodeReady <-> Decoding
                                                                 |   |
                                                            Finishing |
                                                                 v   v
                                                         Finished / Cancelled / Failed
```

An in-flight cancellation uses an explicit `Cancelling` state until its execution fence drains; a
ready-state preemption releases its logical reservation before `Preempted`. The exact M1 states,
events, guards, resource effects, and terminal-emission rules are normative in the
[M1 implementation specification](milestones/m1.md#11-request-finite-state-machine).

`WaitingForRemoteKv` and `TransferringKv` are added in M18. Terminal states are absorbing. Every
transition declares:

- the event that causes it;
- which component may perform it;
- KV reservation/refcount changes;
- whether a response is emitted;
- the timeout/cancellation behavior;
- metrics and trace events.

Keep `num_prompt_tokens`, `num_computed_tokens`, `num_committed_output_tokens`, and
`num_scheduled_tokens` distinct. A scheduled token is not computed until the execution fence succeeds;
a sampled token is not committed until stop/verification logic accepts it.

### 7.3 API semantics

The embeddable API is asynchronous:

```cpp
class AsyncEngine {
 public:
  virtual absl::StatusOr<RequestHandle> Submit(GenerateRequest request) = 0;
  virtual absl::Status Cancel(RequestId id) = 0;
  virtual absl::StatusOr<ResponseEvent> Next(RequestHandle&, Deadline) = 0;
  virtual absl::Status Shutdown(Deadline) = 0;
};
```

`RequestHandle` owns the client's subscription, not the internal request. Dropping it posts
cancellation according to policy. `ResponseEvent` is `TokenDelta`, `UsageUpdate`, or exactly one
terminal `Finished`/`Cancelled`/`Error`. Per-request events are ordered; cross-request ordering is not
guaranteed. Backpressure on a slow consumer is bounded; configurable policy cancels it or coalesces
non-terminal events, never blocks the engine loop.

## 8. Artifact loading, tokenization, and model construction

### 8.1 Artifact pipeline

```text
ModelLocator
  -> ModelManifestReader
  -> HF config/tokenizer metadata validation
  -> SafetensorsIndex + mapped shard files
  -> ModelFactory creates ModelSpec
  -> WeightPlanner maps external names to ParameterSpec and ShardSpec
  -> MemoryPlanner proves weights + workspaces + KV budget fit
  -> WeightLoader stages/checks/transforms/copies each rank's weights
  -> ModelInstance warm-up and health probe
```

Key modules:

- `ModelLocator`: local paths only initially; canonicalizes paths and rejects traversal/special files.
- `ModelManifestReader`: reads `config.json`, generation defaults, tokenizer files, safetensors index,
  declared checksums, revision, and engine compatibility version.
- `SafeTensorReader`: bounds-checks header length, JSON, offsets, dtype, alignment, overlap, and file
  size before returning mapped read-only tensor views.
- `ModelRegistry`: maps a validated architecture tag to one factory; no model-name string checks in
  runtime code.
- `WeightPlanner`: produces a complete one-to-one mapping. Missing, duplicate, unexpected, or
  shape-incompatible weights fail startup. It also describes transpose, concatenation, packing,
  quantization metadata, and TP/PP/EP slicing without performing I/O.
- `MemoryPlanner`: accounts for weights, allocator reserve, KV pool, activation/workspace high-water
  marks, CUDA graphs, NCCL, and a safety margin. Startup fails before partial loading if it cannot fit.
- `WeightLoader`: pipelines memory mapping -> pinned staging -> optional transform -> async H2D copy;
  limits open files and staging bytes; reports per-tensor context on failure.

Do not load PyTorch pickle. Conversion tooling may produce a versioned InferX manifest plus
safetensors, but the server loader remains data-only.

### 8.2 Model identity

`ModelFingerprint` includes architecture/config digest, tokenizer digest, weight revision/checksum
manifest, quantization/layout version, rope/scaling parameters, adapter identity, and parallel shard
identity. Prefix and remote-KV namespaces also include tenant sharing scope. Never hash only token
IDs: identical tokens under different models, RoPE settings, adapters, or tenants are not reusable.

### 8.3 Tokenization and streaming detokenization

```cpp
class Tokenizer {
 public:
  virtual absl::StatusOr<std::vector<TokenId>> Encode(
      std::string_view utf8, const EncodeOptions&) const = 0;
  virtual absl::StatusOr<std::string> Decode(
      std::span<const TokenId>, const DecodeOptions&) const = 0;
  virtual absl::StatusOr<std::unique_ptr<IncrementalDecoder>>
      NewIncrementalDecoder() const = 0;
  virtual const TokenizerMetadata& metadata() const = 0;
};
```

The selected third-party tokenizer must correctly handle tokenizer JSON models, normalization,
pre-tokenization, added/special tokens, byte fallback, invalid UTF-8 policy, BOS/EOS insertion, and
incremental decode. Differential tests use Hugging Face Tokenizers over a multilingual/fuzz corpus.
Chat-template rendering belongs in `PromptProcessor`, before tokenization, and must use a vetted
template implementation or a deliberately supported subset; silently approximating Jinja templates
is forbidden.

Tokenizer errors happen before admission to GPU queues. Tokenization pools have byte and job limits,
and record queue time separately from engine TTFT.

## 9. Model, operator, kernel, and execution boundaries

### 9.1 Model semantics

`ModelSpec` is immutable metadata. A `ModelDefinition` validates and instantiates semantic modules:

```text
LlamaForCausalLM
  TokenEmbedding
  repeated LlamaDecoderLayer
    RMSNorm
    SelfAttention(QKV projection, RoPE, attention, output projection)
    RMSNorm
    SwiGLU MLP(gate/up/down)
  Final RMSNorm
  LM Head
```

Model code receives `ForwardBatch` and an `OpExecutor`; it never chooses CUDA streams, allocates KV
pages, or calls NCCL directly. Model-specific code is limited to configuration/weight mapping and
semantic composition. Shared modules cover embeddings, norms, RoPE, dense linears, attention, MLP,
and later MoE/MLA/sliding-window variants.

### 9.2 Operator contracts and backend registry

Each operation contract describes shapes, layouts, aliases, supported dtypes, scratch requirement,
and asynchronous completion semantics. Important contracts include `GemmOp`, `EmbeddingOp`,
`RmsNormOp`, `RopeOp`, `PagedAttentionOp`, `ActivationOp`, `LogitsOp`, `SamplingOp`, and collective ops.

`KernelRegistry` is frozen after device/model initialization. Its key includes operation, phase
(prefill/decode/mixed), GPU architecture, dtype/quantization, head geometry, page size, layout,
batch/sequence bucket, graph compatibility, and workspace limit. Selection is deterministic and
logged. Forced-backend config supports testing. Unsupported combinations fail during warm-up, not on
the first live request.

The backend order is pragmatic:

1. CPU/reference implementations for small correctness tests.
2. cuBLASLt for production dense GEMM and heuristic selection.
3. FlashInfer/CUTLASS integration where a stable native C++ interface matches the chosen CUDA pin.
4. Small custom CUDA kernels for glue operations and metadata transforms.
5. Custom fused kernels only after profiling proves a gap and correctness/performance gates exist.

`hpc-ops` is experimental: its current upstream focus and Python-facing integration make it a
benchmark/source of candidate SM90 kernels, not a baseline runtime dependency.

### 9.3 Hardware-neutral execution interface

```cpp
class ExecutionBackend {
 public:
  virtual BackendCapabilities capabilities() const = 0;
  virtual absl::StatusOr<ModelHandle> Load(const ModelLoadPlan&) = 0;
  virtual absl::StatusOr<ExecutionTicket> Submit(
      const StepPlan&, const ModelHandle&) = 0;
  virtual absl::StatusOr<StepResult> Poll(ExecutionTicket&) = 0;
};
```

`StepPlan` contains IDs, token spans, positions, sequence lengths, logical block-table snapshots,
sampling work, expected collective sequence, and buffer-slot IDs. It does not contain `cudaStream_t`,
raw device pointers, HTTP values, or owning request pointers. The CUDA backend resolves logical
handles to device metadata and pointers.

### 9.4 CUDA implementation

`platform/cuda` contains:

- `CudaDevice`/`CudaDeviceGuard` and capability discovery;
- move-only `CudaStream`, `CudaEvent`, `CudaGraph`, cuBLASLt and NCCL handle wrappers;
- `CudaAllocator` for startup/large allocations and suballocation pools for serving;
- pinned host pools, event pools, stream roles, metadata ring buffers, and workspace arenas;
- `CudaExecutionBackend`, `CudaModelInstance`, `CudaGraphCache`, and kernel adapters;
- error translation and device-health monitoring.

Use separate compute, collective, and KV-transfer streams only when events express true dependencies.
More streams are not automatically faster. The initial eager path uses one compute stream and no
implicit host synchronization. CUDA graph capture arrives only after stable buffer addresses and
shape buckets exist.

## 10. Scheduler, continuous batching, and resource transactions

### 10.1 Modules

```text
Scheduler
  AdmissionController       validates queue/resource/SLO limits
  SchedulingPolicy          FCFS first; priority/fairness later
  BatchPlanner              token and sequence budgets
  PrefillPlanner            cache hit + prompt chunks
  DecodePlanner             one or speculative token slots
  PreemptionPolicy          recompute victim choice initially
  ResourceAccountant        immutable capacity snapshot
  PlanValidator             debug/test invariant checker
```

The scheduling policy consumes `SchedulingSnapshot` and returns candidates. It cannot mutate requests
or allocate pages. The coordinator then opens a `KvReservationTxn`, reserves exact blocks/workspaces,
builds a `StepPlan`, validates it, and commits the reservation only after the backend accepts the
plan. Submission failure rolls it back. Execution failure invalidates any pages being written before
releasing them.

The minimum plan vocabulary is explicit and versionable:

```cpp
enum class WorkKind { kPrefill, kDecode, kVerify };

struct ScheduledSequence {
  RequestId request;
  SequenceId sequence;
  RequestEpoch epoch;
  WorkKind kind;
  TokenRange input_tokens;
  PositionRange positions;
  KvReadPlan kv_read;
  KvWritePlan kv_write;
  std::optional<SamplingWork> sampling;
};

struct StepPlan {
  StepId id;
  ModelId model;
  std::vector<ScheduledSequence> sequences;
  StepResourceUse resources;
  CollectiveSequence collectives;
  PlanBufferSlot metadata_slot;
};
```

`RequestEpoch` increments on preemption/restart of a logical request. Completion must match step,
request, epoch, metadata slot, and page generations before it can commit output.

### 10.2 Budget and fairness model

Every step enforces:

- `max_scheduled_tokens`, counting uncached prefill and decode/lookahead tokens;
- `max_sequences` and phase-specific sequence limits;
- available KV pages plus configured low-water reserve;
- metadata/workspace/graph bucket capacity;
- per-request maximum context and deadline;
- a bounded prefill share so long prompts cannot indefinitely delay decode;
- per-tenant queue/concurrency/token limits at admission.

FCFS with decode progress protection is the first policy. Priority aging is added with starvation
tests. Scheduling is deterministic for a fixed snapshot and seed. Exact per-step decisions can be
serialized to a replay log for failures and benchmarks.

### 10.3 Continuous and chunked batching

The iteration loop is:

```text
drain request/cancel/completion events
  -> finalize prior step and commit tokens/KV
  -> snapshot requests and capacity
  -> match local/remote prefix where applicable
  -> choose decode work and bounded prefill chunks
  -> reserve resources transactionally
  -> submit immutable plan
  -> publish outputs/metrics while GPU executes
```

Prefill is represented as `[begin_token, end_token)` chunks in the plan. Chunk boundaries align to
KV pages except for the final prompt tail. Decode gets a configurable minimum token share when active.
Initially mixed prefill/decode batches are allowed only if the attention backend advertises support;
otherwise the planner alternates phase-homogeneous steps. The choice must be benchmarked in M8.

Preemption starts with recomputation: release unshared pages, retain immutable shared prefix pages,
move the request to `Preempted`, and requeue with aging. Swap-to-host is deferred until measurements
show it beats recomputation for target contexts/interconnects. Never preempt a request with an
in-flight writer or KV transfer; mark it for preemption after its fence.

## 11. KV cache and memory management

### 11.1 Geometry and responsibilities

At startup, compute page bytes with checked arithmetic from local layers, KV heads, head dimension,
K/V count, tokens/page, dtype, and layout. Create separate pools when incompatible layer groups or
attention layouts require them. Reserve model/workspace/graph/NCCL/safety memory before assigning the
remaining configured budget to KV.

```text
KvCacheManager
  PhysicalBlockPool       fixed device allocation and free slots
  BlockAllocator          free/reserved/ready state and generations
  SequenceBlockTable      logical token page -> physical handle
  PrefixIndex             reusable full-page prefixes
  EvictionPolicy          unpinned reusable pages only
  KvReservationTxn        atomic reserve/commit/rollback
  KvEventLog              allocation/reuse/eviction/transfer events
  KvTransportCoordinator  added in M18
```

Core page states are `Free`, `Reserved`, `Writing`, `Ready`, `TransferIn`, `TransferOut`, and
`Invalid`. A page can be in the free list only at refcount zero, with no pin and no pending fence.
State transition assertions remain enabled in debug/stress builds.

### 11.2 Block tables and leases

Each sequence owns a logical `SequenceBlockTable`. Full-page shared prefixes hold leases to immutable
`Ready` pages. A request's writable tail is exclusive; the first implementation does not share
partial pages, which avoids copy-on-write races. Append reserves the next page before execution and
commits only the number of successfully computed tokens. Rollback invalidates unwritten slots.

Allocator metadata is coordinator-owned CPU memory. The CUDA backend maintains versioned device
mirrors in pooled metadata buffers. A plan records the block-table epoch; delayed completion for an
old epoch is a fatal invariant violation for that replica.

The logical manager contract is shaped around lifecycle operations, not device addresses:

```cpp
class KvCacheManager {
 public:
  KvCapacitySnapshot Snapshot() const;
  absl::StatusOr<PrefixMatch> MatchAndPin(const PrefixLookup&);
  absl::StatusOr<KvReservationTxn> BeginReservation();
  absl::Status CompleteWrite(const KvWriteCompletion&);
  absl::Status AbortWrite(const KvWriteFailure&);
  absl::Status ReleaseSequence(SequenceId);
};

class KvReservationTxn {
 public:
  absl::StatusOr<KvWritePlan> ReserveAppend(const AppendRequest&);
  absl::Status CommitToWriting(ExecutionTicketId);
  void Rollback();  // Idempotent; also runs if an uncommitted transaction is destroyed.
};
```

Only the coordinator calls these methods. `KvWritePlan` contains generational logical handles; the
CUDA adapter resolves them against the pool for the lifetime of the associated execution ticket.

### 11.3 Prefix cache

`PrefixKey` is a rolling collision-resistant digest over page-aligned token IDs and the complete KV
namespace: model fingerprint, cache layout version, rank/shard, RoPE/scaling, adapter, tenant sharing
scope, and any multimodal/position state. Store parent digest and optional token checksum/length so a
collision cannot silently reuse unrelated KV.

The radix/prefix index maps a logical prefix to immutable full pages. Lookup pins the matched path
during admission; insertion occurs only after the page's write fence succeeds. Eviction selects only
unreferenced/unpinned leaves and frees physical pages through the allocator. Cancellation releases
request leases but does not remove reusable prefix ownership. Cache-disabled mode still maintains a
request's chunk-to-chunk block table; it is not a special scheduler architecture.

### 11.4 General GPU memory

- Long-lived arena: weights and fixed KV pools.
- Stable graph arena: graph input/output/metadata addresses by bucket.
- Workspace pool: bounded leases keyed by backend and shape class.
- Metadata ring: pinned host plus device mirrors for in-flight steps.
- Transient device arena: stream-ordered suballocations with event-based reclamation.

Record current/peak/reserved/fragmented bytes per category. Allocation failure is not recovered by
blind retries. Admission shortage is normal and queues/preempts; startup budget failure aborts;
unexpected backend OOM poisons the replica after collecting a memory snapshot.

## 12. Sampling and output processing

Sampling is independent of model execution semantics:

```text
logits -> optional logprobs -> penalties -> temperature -> top-k/top-p/min-p
       -> RNG sample -> stop-token/length decision -> incremental detokenizer -> response event
```

Modules are `SamplingParamsValidator`, `LogitsProcessorChain`, `Sampler`, `RngState`, `StopMatcher`,
`LogProbCollector`, and `OutputAssembler`. Greedy CPU sampling is acceptable only for the first
vertical slice. M9 implements deterministic CPU sampling for the complete supported parameter set;
M12 moves the hot path to GPU and copies only selected tokens/logprobs.

Use a counter-based RNG keyed by `(request seed, sequence, sample index)` so results do not depend on
batch composition or scheduling. Define tie-breaking, NaN/Inf behavior, EOS suppression, min tokens,
and the exact order of penalties in the API contract. Stop strings require incremental UTF-8-aware
matching across token boundaries and must not leak excluded stop text. Cancellation and length/EOS
termination pass through the same finalization path.

## 13. Networking and production request handling

Keep protocol translation outside the engine. `inferx_server` owns Boost.Asio/Beast I/O contexts,
HTTP parsing, JSON validation, authentication hooks, rate limits, deadlines, and SSE encoding. It
maps protocol requests to `GenerateRequest` and engine events back to wire responses.

Initial endpoints are `/health/live`, `/health/ready`, `/metrics`, `/v1/models`,
`/v1/completions`, and `/v1/chat/completions`; the generation endpoints support streaming and
non-streaming responses. Readiness is false until artifacts are validated, weights load, warm-up
passes, and worker health succeeds. Liveness does not claim model readiness.

Production rules:

- Cap request line/header/body sizes, nesting depth, prompt bytes/tokens, connections, in-flight
  requests, and per-tenant work. Enforce timeouts for headers, body, tokenization, queueing,
  generation, writes, and idle keep-alive.
- Validate JSON into typed protocol objects; reject unknown/unsupported generation options rather
  than ignoring them. Convert engine statuses to stable HTTP error codes without leaking paths or
  device internals.
- A disconnected SSE client posts cancellation. Response writes are bounded and occur off the
  coordinator thread. Exactly one `[DONE]` or terminal JSON response is attempted.
- Authentication/authorization is a small interface; deployment may terminate TLS and authenticate
  at a trusted proxy in the first release. Document trusted proxy headers. Do not invent crypto.
- The public HTTP compatibility version is tested against captured OpenAI-compatible requests. API
  compatibility does not require implementing unsupported endpoints.
- Internal worker/control traffic uses protobuf/gRPC once processes are split. Bulk activations and
  KV do not travel through gRPC; they use NCCL/P2P/NIXL data paths referenced by versioned descriptors.

## 14. Observability, reliability, and operations

### 14.1 Metrics

Expose bounded-label counters, gauges, and histograms:

- requests admitted/rejected/completed/cancelled/failed by model and reason class;
- queue depth/time, active sequences, scheduled prefill/decode tokens, preemptions, deadline misses;
- TTFT, inter-token latency/TPOT, end-to-end latency, output-token throughput, request goodput;
- KV pages free/reserved/ready/pinned, allocation failures, prefix queried/hit tokens, evictions,
  transfer bytes/time/failures;
- model load/warm-up time, weight/KV/workspace/graph bytes, metadata-ring occupancy;
- step scheduling/H2D/compute/collective/sampling/detokenization time and CUDA graph hit rate;
- worker/replica health, queue saturation, NCCL/RPC errors, and restart/drain counts.

Metric names, units, and bucket sets are centralized. No request-, user-, or token-valued labels.

### 14.2 Tracing and logging

A sampled request trace has spans for HTTP receive, tokenization, admission wait, prefix lookup,
each prefill chunk, decode steps (aggregated when needed), sampling, detokenization, response write,
collectives, and KV transfer. Propagate trace context over worker RPC. Logging is structured and
rate-limited, with request/model/replica/rank correlation fields. Prompts and generated text are
redacted by default.

### 14.3 Health and failure containment

`HealthManager` aggregates component states: `Starting`, `Ready`, `Draining`, `Degraded`, `Poisoned`,
and `Stopped`. A recoverable request error cannot degrade the replica. Repeated CUDA launch errors,
illegal access, device loss, corrupted allocator invariants, collective timeout, or plan sequence
divergence poison it. A poisoned replica stops admission, fails in-flight requests with a retryable
status when safe, emits a diagnostic snapshot, and exits for an external supervisor to restart.

Use watchdog deadlines around worker heartbeat, GPU-step completion, collective progress, and KV
transfer. Watchdogs report first; they do not free in-use GPU memory or allow a failed rank to proceed.
Core dumps, GPU dumps, and NCU/Nsight profiling are opt-in operational modes with security guidance.

## 15. Distributed architecture and parallelism

### 15.1 Common topology and control model

`ParallelPlan` is validated at startup and fingerprinted with the model. It maps each parameter,
operator, layer, KV shard, and worker role to explicit rank groups. Do not assume every combination is
a simple `DP * PP * TP * EP` product: attention TP and expert groups can overlap differently in MoE
deployments. `ParallelTopology` therefore exposes named groups (`replica`, `tensor`, `pipeline`,
`expert`, `data`, `kv_transfer`) and rank coordinates derived from an explicit placement map.

```cpp
class ProcessGroup {
 public:
  virtual absl::StatusOr<CollectiveTicket> AllReduce(const CollectiveOp&) = 0;
  virtual absl::StatusOr<CollectiveTicket> AllGather(const CollectiveOp&) = 0;
  virtual absl::StatusOr<CollectiveTicket> ReduceScatter(const CollectiveOp&) = 0;
  virtual absl::StatusOr<CollectiveTicket> AllToAllV(const AllToAllVOp&) = 0;
  virtual absl::StatusOr<CollectiveTicket> Send(const SendOp&) = 0;
  virtual absl::StatusOr<CollectiveTicket> Recv(const RecvOp&) = 0;
};
```

The first backend is NCCL, with grouped send/receive where NCCL lacks a direct variable all-to-all.
The coordinator rank plans each step, broadcasts compact metadata, and every rank validates model,
step, and collective sequence IDs. Model execution is local-SPMD in the modest sense that each rank
runs the same worker loop over its placed shard. A placement compiler is explicitly deferred.

### 15.2 Tensor parallelism (TP)

Implement TP first because it lets one dense model span GPUs and has limited scheduler impact.

- `ColumnParallelLinear` shards output features; optionally gathers results.
- `RowParallelLinear` shards input features and reduces outputs.
- Q/K/V and attention heads are sharded with explicit GQA divisibility/replication rules.
- MLP gate/up are column-parallel; down is row-parallel.
- Embedding/LM head may be vocab-parallel; distributed top-k/logprob handles sharded vocab without
  gathering full logits when implemented.
- Each rank owns only its weight and KV-head shard. Block IDs are local; a logical sequence page is a
  vector of rank-local handles coordinated by the leader.

Collectives run on a dedicated stream only when compute dependencies are event-linked. All ranks
must enter collectives even for empty local work. Correctness compares 1 GPU and 2/4/8 GPU logits and
tokens using identical weights, and failure tests kill/stall one rank.

### 15.3 Pipeline parallelism (PP)

PP partitions whole layer ranges and is added after TP is stable. `PipelineStagePlan` owns contiguous
layers, activation buffer geometry, peer ranks, and token-feedback routing. The scheduler forms
microbatches with IDs and tracks stage occupancy; a step is complete only after the last stage samples
and the token/finish decision reaches the leader and any stage needing next-token input.

Start with an inference forward pipeline and a bounded number of in-flight microbatches. Avoid a
training-specific 1F1B abstraction. Activation send/recv, microbatch order, cancellation drain, and
uneven layer partitioning are explicit. PP can reduce cross-GPU bandwidth relative to TP on slow
links but introduces bubbles and complex continuous-batch state; topology recommendations must come
from benchmark data, not a universal default.

### 15.4 Data parallel replicas (DP)

DP means independent model replicas with separate request schedulers and KV pools. `ReplicaRouter`
uses health, admission capacity, estimated prompt/decode cost, and optional prefix-affinity hints;
round-robin is the reference policy. A request is owned by exactly one replica until a later migration
feature. MoE attention-data-parallel layouts are represented by placement groups, not conflated with
request-replica DP.

### 15.5 Mixture-of-experts and expert parallelism (EP)

MoE adds `RouterOp`, deterministic top-k selection, token count/prefix sum, permutation,
`GroupedGemmOp`, weighted combine, and expert placement. EP shards experts across an expert group:

```text
local hidden states -> route/top-k -> pack by destination
  -> AllToAllV dispatch -> local grouped expert GEMMs
  -> AllToAllV combine -> weighted reduction -> shared attention path
```

Correctness requires stable tie-breaking and no token dropping by default. Capacity factors and token
drops are opt-in and reported. Track routing skew, empty/hot experts, bytes per peer, dispatch/combine
time, and load imbalance. Start with NCCL grouped P2P; qualify DeepEP/NVSHMEM or another specialized
backend only behind the same contract. Prefill throughput and decode low-latency paths may choose
different algorithms. Expert parallel load balancing/migration is a later feature and cannot mutate
placement while steps are in flight.

### 15.6 Relationship among TP, PP, EP, and DP

`ParallelPlanValidator` must prove parameter coverage, layer coverage, head/expert divisibility or
replication, compatible KV layout, peer reachability, and identical collective ordering. A supported
combination is listed in a versioned capability matrix; arbitrary mixtures are rejected. Practical
order is TP, then DP routing/worker isolation, then PP, then MoE/EP, and only then heterogeneous
prefill/decode layouts.

## 16. Disaggregated prefill/decode and remote KV

Disaggregation is designed in now but implemented after local scheduling and distributed execution
are mature. Prefill is compute-heavy and decode is commonly memory/latency-sensitive, so separate
pools can use different GPU types and parallel plans. The benefit must exceed routing and KV-transfer
cost for the target workload.

```text
API/GlobalRouter
  -> select DecodeReplica and reserve destination KV
  -> local prefix lookup at decode side
  -> select PrefillReplica for missing prompt suffix
  -> prefill and publish versioned KvDescriptor
  -> transfer into destination pages, possibly with layout mapping
  -> verify completion across all destination ranks
  -> decode; acknowledge source release
```

Key modules are `WorkerDirectory`, `PlacementManager`, `DisaggregatedRouter`, `KvTransferPlanner`,
`KvLayoutMapper`, `KvTransport`, and `TransferSession`. Control messages use gRPC/protobuf; the data
plane starts with CUDA P2P for same-node experiments and NIXL for GPUDirect/RDMA after qualification.

`KvDescriptor` contains schema version, global request ID, model/cache fingerprint, source and target
parallel plans, token interval, layer/head shard mapping, dtype/layout, page size, checksums, source
worker/lease IDs, and destination block generations. It contains no durable raw pointer. The decode
side allocates destination pages before transfer; source pages stay pinned until all target ranks
acknowledge success or timeout cleanup completes.

Transfer state extends the request FSM:

```text
Prefilling -> WaitingForTransfer -> TransferringKv -> DecodeReady
                    |                    |
                    +-> Recompute/Failed <-+
```

KV exchange is independent of `KvCacheManager` allocation and of transport implementation. A layout
mapper handles heterogeneous TP/PP only for explicitly supported pairs and is correctness-tested
against local recomputation. Transfer overlaps other requests' compute, never the source writes or
destination reads for the same pages. A lost completion/duplicate message is handled idempotently by
session and page generation IDs.

Hierarchical KV (`HBM -> peer HBM -> host DRAM -> remote DRAM -> NVMe`) is a subsequent policy layer.
Its eviction/admission compares reuse probability and transfer cost with recomputation cost. Remote
hits are hints until checksums and model/layout namespace match; a timeout falls back to recompute
when the deadline and capacity permit.

## 17. Third-party dependency policy

### 17.1 Rules

- Reuse mature infrastructure and kernels; spend project effort on scheduling, lifecycle, resource
  accounting, KV semantics, execution planning, model integration, and backend selection.
- Pin exact commits/releases in one dependency lock document. CI builds supported CUDA/compiler
  combinations and records licenses/SBOM. Upgrades require correctness and benchmark comparison.
- Keep third-party warnings out of project warning policy without suppressing warnings in InferX.
- Wrap dependencies only at architectural seams (`Tokenizer`, `SafeTensorReader`, `ProcessGroup`,
  `KvTransport`, `MetricsSink`, kernel adapters), not standard containers or smart pointers.
- No source edits inside submodules. Carry a documented patch through CMake or an owned fork only
  after an ADR and upstream issue/PR.
- A declared submodule is not an approved dependency. M0 may remove or defer unused submodules.

### 17.2 Existing declared submodules to qualify

| Dependency | Intended use | Decision/gate |
|---|---|---|
| Abseil | `Status`/`StatusOr`, flags, logging, strings, hashing utilities | Baseline. Verify selected pin supports the chosen compiler/C++23 mode |
| `divedb/tokenizer` | Native tokenizer JSON encode/decode | M0/M3 spike. Keep only if differential fidelity, incremental decode, thread safety, API stability, and license pass |
| Boost.Beast submodule | HTTP/SSE over Boost.Asio | Baseline candidate. Its required Boost.Asio/System headers and build strategy must be pinned explicitly; Beast alone is not a complete dependency story |
| Folly | Possible bounded queues/futures/executors | Deferred. Prefer C++23 + Asio for the initial runtime; remove if no measured need because its transitive/build cost is high |
| CUTLASS | C++ CUDA templates for specialized GEMM/fusions and reference tuning tools | Baseline header dependency after CUDA compatibility qualification; cuBLASLt remains default GEMM |
| FlashInfer | Paged/ragged attention and possible sampling/MoE kernels | M4 spike. Use only a stable native C++ integration compatible with pinned CUDA; do not pull a Python/PyTorch runtime into production |
| Tencent `hpc-ops` | Candidate optimized attention/MoE/sampling/fused communication kernels | Experimental SM90+ backend/benchmark in M12/M16; current requirements and Python-facing packaging make it unsuitable for M0 baseline |

### 17.3 Proposed dependencies

| Dependency | Use | Introduction |
|---|---|---|
| CUDA Runtime/Driver APIs | devices, streams, events, graphs, memory, P2P | M2; system dependency, not vendored |
| cuBLASLt | production FP16/BF16 dense GEMM, heuristics and workspace | M4; CUDA toolkit dependency |
| NCCL | intra/inter-node GPU collectives and grouped point-to-point | M14; system/container dependency, not vendored |
| `syoyo/safetensors-cpp` (candidate) plus the Hugging Face format conformance corpus | safe mapped weight/index reading | M0 security/fuzz/API audit is mandatory; its documented validation gaps mean it is not approved by declaration. If rejected, build a narrow bounds/shape verifier over simdjson and the official format rather than another general serializer |
| simdjson | model/config/manifest and HTTP JSON parsing with size/depth validation | M1/M3/M9 |
| GoogleTest/GoogleMock | unit, integration, death/failure, parameterized tests | M0 |
| Google Benchmark | microbenchmarks and stable benchmark JSON output | M0 |
| RapidCheck (or equivalent property library) | allocator/FSM/scheduler generated operation tests | M6; first prove maintenance/toolchain fit |
| protobuf + gRPC C++ | versioned coordinator/worker and disaggregation control plane | M14/M18; not bulk tensor transport |
| prometheus-cpp | Prometheus metrics exposition | M10 |
| OpenTelemetry C++ | tracing and context propagation/export | M10 |
| NIXL | GPUDirect/RDMA KV data plane | M18, optional plugin after hardware and license qualification |
| BLAKE3 C/C++ implementation | fast model/prefix/page fingerprints | M6/M11; collision guard includes length/check data |

Potential dependencies must not be added “just in case.” gRPC, OpenTelemetry, NIXL, RapidCheck, and
specialized kernel libraries remain feature-gated so a minimal embedded engine has a small closure.
Do not add a second JSON, logging, status, HTTP, RPC, or metrics stack without an ADR.

## 18. Verification strategy and definitions

### 18.1 Test categories

- **Unit:** CPU-only deterministic tests for values, parsers, FSM transitions, policies, allocators,
  layout math, and adapters; CUDA kernel unit tests where relevant.
- **Integration:** boundaries among artifact loader, tokenizer, model, scheduler, KV, executor, server,
  and distributed workers.
- **Correctness:** independent reference results. Python Transformers/Tokenizers and simple FP32 CPU
  implementations are test tools, not runtime dependencies.
- **Stress:** long randomized sequences of submit/cancel/preempt/allocate/free/shutdown operations,
  saturation runs, long-context runs, and multi-hour soaks in nightly CI.
- **Failure:** malformed artifacts/input, injected allocation/launch/RPC/collective/transfer failures,
  worker death, timeout, stale generations, and restart/drain behavior.
- **Performance:** micro, model, scheduler-simulation, and serving benchmarks with machine, GPU,
  clocks/power mode, driver, CUDA, dependency pins, config, model, and workload recorded.

Every milestone table explicitly covers all six categories. “N/A” requires a reason in the milestone
PR; it is not silently omitted.

### 18.2 Correctness thresholds

- CPU metadata/allocator/FSM tests require exact equality and invariant checks.
- FP32 reference operators use operation-specific `atol`/`rtol` documented beside tests.
- FP16/BF16 model logits compare max/mean error and top-k agreement against a fixed Transformers
  reference. M5 sets numeric thresholds from an empirical error study; they are not loosened to make
  failures pass.
- Greedy generation must match token-for-token for fixed fixtures. Sampling compares deterministic
  engine paths with fixed counter RNG and uses distribution tests for reference algorithms.
- Parallel and graph/optimized paths compare against the accepted single-GPU eager baseline.
- Quantized models use per-format quality criteria on a fixed evaluation subset as well as kernel
  numeric tolerances.

### 18.3 Performance definitions

Report prompt tokens/s, output tokens/s, requests/s, GPU utilization, memory high-water mark, TTFT,
TPOT/inter-token latency, end-to-end latency at p50/p90/p95/p99, and goodput: maximum request rate
meeting declared TTFT and TPOT SLOs. Use open-loop arrival tests for serving claims and closed-loop
tests only when labeled. Include prompt/output length distributions, prefix reuse, sampling config,
and concurrency.

Gates are baseline-relative until M0 records the reference hardware. Kernel changes may not regress
their target shape suite by more than 3% median without an ADR; end-to-end changes may not regress
goodput or p99 by more than 5% on the reference workload. A claimed optimization must show at least a
5% end-to-end improvement or a needed capability, not only a favorable microbenchmark.

### 18.4 CI tiers

1. Presubmit CPU: format, configure/build, header self-containment, clang-tidy/IWYU, unit tests,
   ASan/UBSan, artifact fuzz smoke tests.
2. Presubmit GPU: one supported GPU, CUDA unit/integration tests, tiny-model correctness, compute-
   sanitizer smoke tests.
3. Nightly GPU matrix: supported SM/CUDA pins, full correctness sweeps, TSan CPU tests, randomized
   stress/failure tests, performance comparison.
4. Nightly distributed: 2/4/8 GPUs where available, rank-failure tests, topology and collective tests.
5. Release: multi-hour soak, public API conformance, dependency/SBOM/license/security scan,
   reproducible container build, and signed benchmark/correctness report.

## 19. Incremental implementation roadmap

### 19.1 Dependency graph and release cuts

```text
M0 Repository/toolchain
 |
 M1 Core contracts + simulator
 |\
 | M3 Artifacts + tokenizer
 M2 CUDA/tensor/memory
 |/
 M4 Operators/backends
 |
 M5 Minimal single-request E2E
 -> M6 Paged KV
 -> M7 Async engine + continuous batching
 -> M8 Chunked prefill + robust scheduling
 -> M9 Sampling + HTTP
 -> M10 Observability/reliability             [developer preview]
 -> M11 Prefix caching
 -> M12 CUDA graphs/overlap/GPU sampling      [single-GPU v0.1]
       |\
       | M13 Quantization + second dense model
       M14 Worker isolation + tensor parallelism
         -> M15 Data/pipeline parallelism     [distributed dense v0.2]
             |\
             | M16 MoE + expert parallelism
             | M18 Disaggregated P/D + remote KV
             M17 Speculative decoding
```

M13 and M17 are feature tracks after the stable single-GPU core and can proceed independently.
M18 depends on M11 prefix identity/leases, M14 worker protocols/TP, and M15 topology/PP support. M16
depends on M14 collectives and the placement model, but not on M18.

Every milestone must also update documentation, configuration schema, capability matrix, changelog,
and benchmark manifest. A milestone is complete only when all listed gates run in CI or a documented
hardware-constrained release job.

### M0 — Reproducible repository and dependency qualification

**Depends on:** nothing.

**Implementation specification:** [`docs/milestones/m0.md`](milestones/m0.md)

**Deliverables**

- Top-level CMake project, presets for CPU debug/sanitizers and CUDA release, install/export targets,
  and `inferx::` target naming.
- `cxx_std_23`, `.clang-format` corrected to C++23, clang-tidy/IWYU/format targets, warnings-as-errors
  for InferX, generated `compile_commands.json`, and header self-containment checks.
- GoogleTest and Google Benchmark; skeleton libraries/directories from section 4; a trivial CLI.
- Dependency lock, license inventory, SBOM procedure, supported compiler/CUDA/driver/SM matrix, and
  container/dev setup. Initialize and qualify declared submodules; record keep/defer/remove decisions.
- ADRs for initial platform/model/artifact scope, error model, dependency mechanism, and exception
  boundary. CI tier 1 plus a GPU smoke job when hardware exists.

**Tests**

- Unit: a test target links `base`; every public header compiles alone.
- Integration: fresh-clone configure/build/test/install and downstream `find_package(inferx)` smoke.
- Correctness: compile-time checks confirm C++23 features and strong compiler settings.
- Stress: repeat configure/build/test with parallel jobs; no undeclared generated-file dependency.
- Failure: missing CUDA/dependency and unsupported compiler presets fail with actionable diagnostics.
- Performance: record clean and incremental build time/binary sizes; no pass threshold yet because this
  establishes the baseline.

**Completion criteria**

- A fresh checkout plus documented submodule command builds and passes CPU CI with one preset; the GPU
  preset builds a kernel and detects device capability where a runner is available.
- Format, tidy, header, ASan, and UBSan jobs are green with zero project warnings.
- Every dependency has an exact pin, license, owner, use, feature flag, and upgrade procedure. No
  uninitialized submodule is a silent required build input.

### M1 — Core values, configuration, request FSM, and scheduler simulator

**Depends on:** M0.

**Implementation specification:** [`docs/milestones/m1.md`](milestones/m1.md)

**Deliverables**

- Status conventions, strong IDs, checked counts/ranges/bytes, clocks, immutable config pipeline, and bounded
  queue contract.
- Hardware-neutral `GenerateRequest`, `RequestContext`, transition table, `SchedulingSnapshot`,
  `StepPlan`, resource counters, FCFS policy, and deterministic simulator with fake execution latency.
- Invariant checker and event/replay log format. No tensor kernels or CUDA.

**Tests**

- Unit: every valid/invalid FSM transition, config constraint, checked overflow, budget, and FCFS tie.
- Integration: submit -> simulate prefill/decode -> terminal response, including cancellation/deadline.
- Correctness: replaying identical input produces byte-identical plan/event logs.
- Stress: at least 1,000,000 randomized lifecycle/scheduling operations under ASan/UBSan and nightly
  TSan, with all invariants checked.
- Failure: queue full, duplicate ID, impossible context, stale completion, invalid transition, and
  injected executor failure reach the documented terminal state without leaked capacity.
- Performance: scheduler benchmark at 1/32/256/1024 active requests; track allocations and p50/p99.

**Completion criteria**

- Resource totals return exactly to baseline after every generated scenario and shutdown state.
- At 1,024 active synthetic requests, planning p99 is <= 1 ms on the recorded reference CPU and makes
  zero heap allocations after simulator warm-up; otherwise an ADR revises the data structures/gate.
- The transition table and `StepPlan` schema are documented and used by code, not duplicated prose.

### M2 — Device-neutral tensors and CUDA runtime/memory layer

**Depends on:** M1.

**Implementation specification:** [`docs/milestones/m2.md`](milestones/m2.md)

**Deliverables**

- `DType`, `Shape`, `Strides`, owning `Buffer`, tensor views, and CPU allocator.
- CUDA capability discovery, RAII device/stream/event/handles, device and pinned allocations, async
  copies, event pool, metadata ring, workspace arena, and `CompletionFence`.
- CUDA errors translated to status; fake allocator/failure hooks; memory accounting by category.

**Tests**

- Unit: shape/stride/alias/bounds arithmetic and move/destruction semantics; CUDA device/stream/event
  ordering and guard restoration.
- Integration: pinned H2D -> CUDA test kernel -> D2H on asynchronous streams with event dependencies.
- Correctness: randomized copies/strides compare byte-for-byte with CPU; multi-device selection where
  available.
- Stress: 100,000 pooled lease/release cycles and metadata-ring wraparound; compute-sanitizer nightly.
- Failure: allocation exhaustion, invalid device, launch error, stale fence, and teardown with pending
  work follow documented containment and leak no owned allocation.
- Performance: pageable vs pinned bandwidth, allocation/pool latency, event overhead, and test-kernel
  launch latency recorded by GPU/driver/CUDA pin.

**Completion criteria**

- Sanitizers and compute-sanitizer report no leak, race, invalid access, or unintended host sync.
- Pool steady state performs zero CUDA allocations and zero host heap allocations per lease cycle.
- Measured bandwidth/latency baselines are stored as benchmark JSON; wrapper overhead for event/copy
  submission is <= 3% versus the direct CUDA microbenchmark.

### M3 — Safe model artifacts and tokenizer fidelity

**Depends on:** M1; can run in parallel with M2.

**Deliverables**

- Local `ModelLocator`, manifest/config parser, safetensors/index reader, mapped-file RAII, Llama
  `ModelSpec`, parameter/weight/shard plans, model fingerprint, and inspection CLI.
- Qualified tokenizer adapter, special-token metadata, incremental decoder, prompt processor, and
  bounded tokenization pool. Python scripts generate deterministic tiny Llama artifacts/reference
  fixtures for tests only.
- Strict schema/semantic validation and contextual diagnostics for every artifact.

**Tests**

- Unit: JSON/config fields, tensor headers/offsets/dtypes, weight-name mapping, tokenizer special
  tokens, incremental byte boundaries, and fingerprint namespace changes.
- Integration: load a sharded tiny artifact, map every parameter exactly once, tokenize/decode it,
  and feed a fake model instance.
- Correctness: 100% token-ID and decoded-byte agreement with Hugging Face Tokenizers on at least
  10,000 multilingual, whitespace, special-token, byte-fallback, and randomized fixtures within the
  explicitly supported tokenizer feature set.
- Stress: fuzz safetensors/config/tokenizer inputs and repeatedly map/unmap many shards under file-
  descriptor and staging-byte limits.
- Failure: truncated/overlapping/oversized tensors, integer overflow, symlink/path escape, checksum
  mismatch, missing/extra/duplicate/shape-wrong weights, invalid UTF-8 policy, and unsupported
  tokenizer construct fail safely.
- Performance: tokenizer throughput/latency by input size and mapped artifact scan/load throughput;
  establish baseline without a pass threshold.

**Completion criteria**

- The deterministic tiny Llama artifact passes complete mapping with no ignored weights.
- Fuzz jobs execute at least 10 million parser operations without crash, OOB access, unbounded
  allocation, or hang before release; presubmit runs a smaller fixed-time corpus.
- The tokenizer dependency decision is closed. If the declared tokenizer fails, replace it through
  the same interface rather than maintaining known divergence.

### M4 — Reference operators and production CUDA backend adapters

**Depends on:** M2 and M3.

**Implementation specification:** [`docs/milestones/m4.md`](milestones/m4.md)

**Deliverables**

- CPU FP32 reference embedding, RMSNorm, RoPE, causal MHA/GQA, SwiGLU, residual, GEMM, and logits.
- cuBLASLt GEMM adapter with algorithm/workspace cache; CUDA glue kernels; FlashInfer native C++
  integration decision and, if accepted, prefill/decode attention adapter.
- Frozen `KernelRegistry`, capability queries, forced backend selection, workspace requirements, and
  warm-up validation. A simple owned CUDA attention fallback exists for supported tiny test shapes if
  FlashInfer is not viable.

**Tests**

- Unit: each op's shapes/layouts/aliases and backend capability/selection matrix.
- Integration: compose one decoder layer on CPU and CUDA, including KV append and read.
- Correctness: dtype/shape/sequence/head/page sweeps against FP32 reference; adversarial NaN/Inf and
  non-contiguous metadata cases where supported.
- Stress: repeated mixed-shape dispatch/workspace reuse and multi-stream execution under compute-
  sanitizer.
- Failure: unsupported SM/dtype/layout, cuBLAS/attention launch failure, insufficient workspace, and
  registry miss fail during warm-up or return scoped status.
- Performance: compare GEMM with direct cuBLASLt and attention with the qualified FlashInfer/backend
  across target prefill/decode shapes; profile representative outliers.

**Completion criteria**

- All combinations required by the M5 Llama configuration have a warm-up-validated backend and meet
  documented numeric tolerances.
- Adapter overhead is <= 3% versus the same underlying library call; no owned custom kernel is
  accepted without a benchmark and independent reference test.
- The FlashInfer/CUDA compatibility ADR is closed with exact supported pins and a fallback plan.

### M5 — Minimal single-request Llama end-to-end path

**Depends on:** M3 and M4.

**Deliverables**

- `LlamaForCausalLM`, GPU weight loading, memory plan, eager `CudaExecutionBackend`, and one-request
  executor with a simple contiguous KV cache.
- Token-ID and text CLI: load -> warm up -> prefill -> greedy decode -> incremental text output.
- Independent reference fixture generation and intermediate-tensor diagnostic mode.

**Tests**

- Unit: Llama config/weight variants, position/RoPE math, layer orchestration, logits extraction.
- Integration: load deterministic tiny model and an opt-in public Llama-compatible checkpoint,
  generate from token and text inputs, unload cleanly.
- Correctness: CPU reference intermediate tensors for tiny model; Transformers logits tolerance and
  token-for-token greedy output for batch 1, varied prompt lengths, EOS, and max length.
- Stress: 1,000 sequential generations with varied legal lengths and repeated load/unload nightly.
- Failure: model-too-large plan, device OOM, malformed prompt, unsupported context, launch error, and
  cancellation between eager steps clean up all memory/state.
- Performance: record load time, first/steady prefill tokens/s, decode TPOT, peak memory, and kernel
  breakdown; no optimization gate yet.

**Completion criteria**

- Greedy output matches the reference for every committed fixture (at least 100 prompts and 32 output
  tokens unless EOS); logits meet thresholds established and documented from M4 error data.
- After 1,000 requests, all non-model memory categories return to their warm baseline and sanitizer/
  profiler checks show no invalid access or unintended per-token `cudaMalloc`/device synchronize.
- One documented CLI command reproduces the path from a fresh supported environment.

### M6 — Production paged KV cache

**Depends on:** M5.

**Deliverables**

- Fixed `PhysicalBlockPool`, generational allocator, sequence block tables, transactional reservation,
  device block-table mirrors, paged attention metadata, and configurable page geometry.
- Contiguous KV retained as a correctness-only backend. Memory budget and low-water admission signal.
- Debug invariant dump and deterministic allocator operation replay.

**Tests**

- Unit: every page transition, lease/refcount, generation rejection, append/commit/rollback, boundary
  position, and memory formula.
- Integration: replace contiguous KV in M5 and decode variable-length sequences through page
  boundaries and pool pressure.
- Correctness: paged vs contiguous logits/tokens for MHA/GQA, page sizes, prompt/decode lengths, and
  partial final pages.
- Stress: >= 1,000,000 randomized allocate/share-free/rollback operations and GPU saturation with
  frequent page reuse; verify allocator after each operation in a debug run.
- Failure: exhausted pool, stale block/epoch, failed KV write, double release, cancellation, and
  shutdown with in-flight pages never expose invalid data or reuse early.
- Performance: allocation/plan cost, attention latency by page size/context distribution, metadata
  transfer, peak usable KV tokens, and fragmentation.

**Completion criteria**

- Paged and contiguous paths satisfy the same correctness suite with no token divergence.
- Internal fragmentation is less than one page per active sequence plus documented pool/layout
  padding; no serving-step device allocation occurs.
- The page-size/layout ADR is decided from correctness, capacity, and target-shape benchmark data.

### M7 — Asynchronous request engine and continuous batching

**Depends on:** M6.

**Deliverables**

- `AsyncEngine`, coordinator loop, request registry/handles, bounded channels, device worker, execution
  tickets, ordered response events, streaming, basic cancellation/deadline handling, and graceful
  shutdown.
- Iteration-level FCFS continuous batching with full-prompt prefill and one-token decode, exact token/
  sequence/KV budgets, and deterministic plan replay.
- Tokenizer pool integrated ahead of admission; CPU scheduling/output work overlaps asynchronous GPU
  execution where dependencies allow.

**Tests**

- Unit: channel/backpressure, response ordering, completion epochs, admission accounting, and
  shutdown transitions.
- Integration: concurrent text/token requests enter/leave batches at different iterations and stream
  through the embedded API.
- Correctness: batched outputs/logits match serial M5 execution across varied arrivals, lengths, and
  batch composition.
- Stress: thousands of concurrent submissions, cancellation storms, slow consumers, queue
  saturation, and two-hour nightly soak.
- Failure: dropped handle, timeout in every state, stale/duplicate completion, worker submission
  failure, and forced shutdown produce one terminal event and release all resources.
- Performance: concurrency sweeps measure throughput/TTFT/TPOT/p99, coordinator planning time, CPU
  utilization, and H2D/compute overlap against serial M5.

**Completion criteria**

- No request starves under FCFS; completed+canceled+failed equals admitted; every request has exactly
  one terminal event; all queue/KV/workspace counts return to baseline.
- Continuous batching improves output-token throughput by at least 2x at a suitable recorded
  concurrency over serial M5 on the reference GPU without >10% p99 TPOT regression for the decode-
  only workload.
- Coordinator p99 loop overhead excluding GPU wait remains <= 1 ms at 1,024 active requests.

### M8 — Chunked prefill, fairness, preemption, and robust scheduling

**Depends on:** M7.

**Deliverables**

- Page-aligned chunked prefill, decode progress reservation, mixed/alternating phase capability,
  priority aging, deadline-aware admission, recompute preemption, low-water handling, and policy
  configuration snapshots.
- Two-phase plan/reservation commit fully wired; exact cached/computed/scheduled/committed counters.
- Scheduler simulation workload importer and policy comparison tool.

**Tests**

- Unit: chunk math/tails, priority aging, victim selection, rollback, low-water, and budget interaction.
- Integration: long prefills interleave with decode; preempt/recompute retains only safe prefix/pages;
  cancellation races with chunk completion.
- Correctness: chunked vs unchunked and preempted vs uninterrupted logits/tokens for boundary-heavy
  inputs and all supported page sizes.
- Stress: adversarial long/short and priority mixes under near-full KV, cancellation/deadline storms,
  and randomized schedules with invariant checks.
- Failure: plan rejection after reservation, execution failure mid-chunk, no allocatable victim,
  impossible deadline, and stale preempted completion recover without leaks/deadlock.
- Performance: chunk-size and token-budget sweeps compare mixed vs alternating phases using goodput,
  TTFT, TPOT p99, scheduler cost, and GPU utilization.

**Completion criteria**

- No eligible request is starved in a 30-minute adversarial run; priority/deadline behavior matches
  documented policy; all correctness fixtures are token-identical.
- For the declared mixed long-prefill/decode workload, selected defaults improve p99 TPOT by >= 20%
  over unchunked prefill without reducing SLO goodput by >5%; otherwise document why chunking is
  capability-only and retain data-driven defaults.
- Reservation rollback and preemption property tests return every resource counter exactly.

### M9 — Complete baseline sampling and OpenAI-compatible HTTP serving

**Depends on:** M8.

**Deliverables**

- Deterministic CPU reference support for temperature, top-k, top-p, min-p if exposed, repetition/
  frequency/presence penalties, seed, EOS/min/max tokens, stop tokens/strings, and requested logprobs.
- Boost.Asio/Beast server, simdjson protocol parsing, SSE and non-streaming responses, endpoint/limit/
  timeout/auth hooks, health/readiness, and protocol-to-engine error mapping.
- CLI/config documentation and a load generator that supports open-loop arrivals.

**Tests**

- Unit: sampling validation/order/ties/RNG, stop matching across UTF-8/token boundaries, JSON and SSE
  encoding, rate/limit accounting.
- Integration: HTTP text/chat requests -> tokenization -> engine -> streaming/non-streaming output;
  disconnect posts cancellation.
- Correctness: fixed-seed sampling matches the independent algorithm fixture; HTTP golden/conformance
  suite checks fields, usage, errors, finish reasons, chunks, and `[DONE]`.
- Stress: >=1,000 concurrent keep-alive/SSE clients, slow readers, malformed-input fuzzing, rate-limit
  saturation, and repeated reconnects.
- Failure: oversized/truncated JSON, unsupported fields, tokenizer/model failure, queue full, deadline,
  disconnect, write failure, and shutdown map to stable terminal protocol behavior.
- Performance: HTTP vs embedded overhead, parser/serialization cost, CPU sampling cost by vocabulary/
  batch, TTFT/TPOT under open-loop load.

**Completion criteria**

- All documented endpoints and parameters pass the versioned compatibility suite; unsupported input
  is rejected explicitly.
- No slow/disconnected client blocks coordinator progress or causes unbounded memory; per-connection
  and global limits hold in stress tests.
- Network/protocol overhead adds <= 5% median end-to-end latency for local batch-1 requests and <= 3%
  throughput loss at the reference saturation point versus embedded API.

### M10 — Observability, failure containment, and developer preview

**Depends on:** M9.

**Deliverables**

- Metrics, tracing, structured/redacted logging, health aggregation, worker/GPU watchdogs, diagnostic
  snapshots, drain/poison behavior, signal handling, and operational runbook.
- Fault-injection framework at allocator, kernel, worker, queue, and response boundaries; reproducible
  request/scheduler trace capture.
- Reproducible production container, dependency/license/SBOM scan, compatibility report, and a
  documented benchmark workload/SLO.

**Tests**

- Unit: metric units/labels/buckets, trace propagation/sampling, redaction, health transitions, and
  watchdog timers.
- Integration: one traced request correlates HTTP/tokenizer/scheduler/GPU/output spans; readiness and
  drain reflect actual model/worker state.
- Correctness: metrics reconstruct admitted and terminal counts exactly; telemetry on/off produces
  identical tokens and request outcomes.
- Stress: telemetry exporter outage/backpressure, log storms, two-hour presubmit-equivalent soak and
  24-hour release soak at varied load.
- Failure: injected CUDA/context/worker/watchdog/exporter failures produce documented containment,
  diagnostics, exit status, and no false readiness.
- Performance: telemetry enabled/disabled comparison for CPU, memory, throughput, and p99 latency.

**Completion criteria**

- All fault scenarios have a bounded terminal outcome; poisoned workers cannot accept work; ordinary
  request errors do not degrade readiness.
- Telemetry adds <= 2% throughput and <= 2% p99 latency overhead with production sampling/export
  settings, and exporter failure cannot stall inference.
- The 24-hour soak has zero unexpected request loss, invariant violation, memory growth beyond 1% of
  fixed pools, or crash. Publish the developer-preview report and known limitations.

### M11 — Cross-request prefix caching

**Depends on:** M10 and M6.

**Deliverables**

- Namespace-complete `PrefixKey`, radix/prefix index, full-page insert/match/pin/unpin, prioritized LRU
  leaf eviction, cache events/metrics, tenant isolation, and cache reset/debug APIs.
- Scheduler uses matched-token cost and pins lookup results through reservation. Cache-disabled path
  remains feature-equivalent except for cross-request reuse.
- Optional deterministic cache-aware waiting-queue policy behind config; FCFS remains reference.

**Tests**

- Unit: split/merge/match, page alignment, refcounts/pins, LRU priority, namespace differences,
  collision guard, and eviction eligibility.
- Integration: shared system prompts, multi-turn-style prefixes, simultaneous same-prefix arrivals,
  cancellation, and eviction/recompute under pressure.
- Correctness: cache on/off logits and tokens are identical at 0/25/50/75/90% reuse; different
  model/revision/RoPE/adapter/tenant scopes never share.
- Stress: millions of randomized tree/allocator operations, high-churn prefix workloads, collision
  injection, and concurrent request lifecycle events through the coordinator.
- Failure: failed page write is never inserted; stale page generation, corrupted checksum, OOM during
  pin/reservation, and reset during drain fail safely.
- Performance: match/insert/evict CPU cost, hit-token rate, saved prefill work, TTFT/goodput, memory
  overhead, and cache-aware vs FCFS policy.

**Completion criteria**

- Zero cross-namespace reuse in the isolation suite and exact cache-on/off output equality.
- Prefix operations have <= 100 microseconds p99 at 100,000 indexed pages on the reference CPU, or a
  benchmark-backed ADR changes the index/design.
- At >=75% reusable prompt tokens, reference workload TTFT improves >=30% and no-reuse throughput
  regresses <=3%; otherwise the feature remains off by default pending redesign.

### M12 — Single-GPU performance: GPU sampling, overlap, and CUDA graphs

**Depends on:** M11.

**Deliverables**

- GPU fused/qualified sampling path with counter RNG and minimal D2H output; asynchronous output
  processing; double/triple-buffered step metadata; CPU plan N+1 overlap with GPU step N.
- CUDA graph cache for measured decode buckets with stable graph arenas, padding policy, capture-safe
  backends, miss/fallback path, memory budget, and hit/eviction telemetry.
- Profile-guided fusions/backend selection and autotune cache keyed by device/software/model/shape.

**Tests**

- Unit: graph bucket/padding/address leases, RNG indexing, autotune key invalidation, and overlap epoch
  protection.
- Integration: eager/graph and CPU/GPU sampling switch per step; fallback for unsupported shape;
  cancellation while later CPU work is prepared.
- Correctness: graph vs eager and GPU vs CPU sampling/logprobs for all parameters and batch/sequence
  buckets; scheduling overlap does not change tokens.
- Stress: graph cache churn, metadata ring saturation, thousands of dynamic batch transitions, and
  compute-sanitizer race checks.
- Failure: capture/replay failure, graph OOM, stale stable buffer, sampling kernel failure, and autotune
  cache corruption fall back or poison according to policy without reuse races.
- Performance: graph hit rate, CPU launch gaps, D2H bytes, sampling latency, eager/graph goodput and
  TPOT for batch buckets; NCU/Nsight profiles for remaining top bottlenecks.

**Completion criteria**

- Selected workload graph hit rate is >=90%; graph arena remains within configured budget; eager
  fallback is always correct.
- GPU sampling reduces D2H logits traffic by >=99% and is not slower than CPU sampling at target
  concurrency; optimized engine improves SLO goodput >=15% over M11 with no correctness regression.
- Publish single-GPU v0.1 artifacts, benchmark report, supported capability matrix, and 24-hour soak.

### M13 — Quantization and second dense model family

**Depends on:** M12. Independent of distributed milestones.

**Deliverables**

- Quantization descriptors/scales in artifact and operator contracts. Add one format at a time:
  FP8 on supported hardware, then one widely used weight-only INT4 format (AWQ or GPTQ selected by
  artifact availability and backend support). KV quantization is a separate experiment/ADR.
- Qwen dense model support by reusing common modules and adding only configuration/weight/semantic
  differences. Quantized backend/capability entries and conversion/inspection tools.

**Tests**

- Unit: scale/zero-point/group/block metadata, packing, shard boundaries, Qwen config and weight map.
- Integration: load/generate FP16/BF16 and each quantized format for Llama and Qwen where supported.
- Correctness: dequantized kernel reference, logits/top-k agreement, greedy fixtures, and a fixed
  quality/perplexity subset with format-specific thresholds set before optimization.
- Stress: repeated quantized load/unload, odd shard/group sizes, long context, and mixed model startup.
- Failure: missing/mismatched scales, unsupported SM/layout, corrupt packed data, memory-plan error,
  and unavailable kernel fail before serving.
- Performance: load time, memory, prefill/decode goodput and accuracy tradeoff versus BF16.

**Completion criteria**

- No duplicated model-runtime architecture logic beyond documented semantic differences.
- Each enabled format meets its predeclared quality threshold, reduces weight memory as theoretically
  expected within 5% metadata overhead, and improves a target capacity/performance metric by >=10%.
- Unsupported model/format/hardware combinations are rejected at validation/warm-up.

### M14 — Worker process isolation and tensor parallelism

**Depends on:** M12.

**Deliverables**

- One process per GPU worker, versioned coordinator/worker protobuf protocol, capability/health
  handshake, shared-memory or compact IPC for metadata where justified, and external supervisor
  integration.
- `ParallelTopology`/`ParallelPlan`, NCCL `ProcessGroup`, collective sequence validation, TP weight/KV
  sharding, column/row parallel linear, head/vocab rules, and distributed sampling path.
- Single-process single-GPU remains supported through the same logical execution contract.

**Tests**

- Unit: rank-group construction, placement coverage, shard slicing/divisibility, protocol versions,
  collective plan ordering.
- Integration: 1/2/4/8-GPU TP load/prefill/decode/cancel/shutdown and worker restart between requests.
- Correctness: per-layer/logit/token equivalence to single-GPU for supported TP sizes, uneven GQA
  rejection/replication rules, and distributed top-k/logprob.
- Stress: repeated group create/destroy, high continuous-batch load, IPC saturation, and multi-hour
  collectives.
- Failure: rank startup mismatch, protocol/model fingerprint mismatch, stalled/killed rank, NCCL
  async error/timeout, duplicate collective sequence, and coordinator death drain the whole replica.
- Performance: TP scaling efficiency, collective/compute overlap, bytes, memory per rank, TTFT/TPOT/
  goodput by link topology and TP degree.

**Completion criteria**

- TP tokens match single-GPU fixtures; model/KV memory is sharded as planned within 5% overhead.
- On a model/shape that cannot or should not run on one GPU, 2/4/8-GPU results publish scaling and
  link-topology data; supported degrees achieve >=70% strong-scaling efficiency from TP2 to TP4 on
  the chosen NVLink reference or are not advertised as performance configurations.
- Killing/stalling any rank yields bounded replica failure, one terminal result per request, and no
  surviving rank continuing mismatched collectives.

### M15 — Data-parallel routing and pipeline parallelism

**Depends on:** M14.

**Deliverables**

- Independent DP replicas and `ReplicaRouter` with round-robin reference plus load/capacity and
  optional prefix-affinity policy; health-aware draining and per-replica telemetry.
- PP contiguous stage placement, activation channels, microbatch IDs/buffers, pipeline-aware scheduler,
  token feedback, uneven partition cost model, and supported TP+PP combinations.
- Topology recommendation tool based on model memory, interconnect, prompt/output distribution, and
  benchmark results.

**Tests**

- Unit: routing ownership/fairness, health removal, stage coverage, microbatch ordering, buffer leases,
  and TP+PP group validation.
- Integration: DP2 routing; PP2/PP4 and supported TPxPP prefill/decode/stream/cancel/drain.
- Correctness: routed replica and PP logits/tokens equal single-replica TP baseline; feedback tokens
  and finish reasons remain ordered through microbatches.
- Stress: skewed arrivals, replica drain/rejoin between requests, pipeline saturation/bubbles,
  cancellation in each stage, and multi-hour mixed-length load.
- Failure: dead/unready replica, activation send timeout, stage death, out-of-order/duplicate
  microbatch, and partial group startup fail/reroute only where request ownership permits.
- Performance: DP throughput scaling and imbalance; PP bubble fraction, activation bandwidth,
  TTFT/TPOT/goodput by microbatch/stage/TP configuration.

**Completion criteria**

- DP2 delivers >=1.8x SLO goodput over one identical replica on the reference workload with balanced
  load and no cross-replica request/KV ownership.
- All advertised PP/TP combinations pass equivalence and failure suites. The recommendation tool does
  not select PP unless measured goodput/memory feasibility beats supported alternatives for the input
  topology.
- Publish distributed dense v0.2 with a complete capability/topology matrix.

### M16 — Mixture-of-experts and expert parallelism

**Depends on:** M14; use M15 topology/routing when available.

**Deliverables**

- One MoE architecture (chosen by available legal test artifacts), router/top-k, pack/unpack, grouped
  GEMM, weighted combine, expert placement, and dense fallback reference.
- EP dispatch/combine over NCCL grouped P2P, no-drop default, deterministic routing, imbalance
  telemetry, and separate prefill/decode algorithms. Optional specialized backend spike behind
  `ProcessGroup`/MoE op contracts.

**Tests**

- Unit: top-k ties, routing weights, prefix sums/permutation/inverse, expert placement, empty/hot
  experts, AllToAllV counts/offsets.
- Integration: single-GPU MoE, EP2/4/8, and supported TP+EP/DP+EP combinations under continuous
  batching.
- Correctness: router choices, per-expert outputs, logits, and tokens match independent dense-dispatch
  reference; no token loss/duplication; quantized MoE only after its own thresholds.
- Stress: adversarial routing skew, empty experts, maximum tokens, repeated collective buffers, and
  long soak.
- Failure: count mismatch, peer/rank loss, grouped GEMM failure, unsupported expert divisibility, and
  capacity overflow fail the replica/request without silent token drops.
- Performance: route/pack/dispatch/GEMM/combine breakdown, link bytes, imbalance, prefill/decode
  goodput, and NCCL vs qualified specialized backend.

**Completion criteria**

- EP output is token-identical to reference greedy fixtures; conservation asserts exactly `tokens *
  top_k` routed assignments before and after communication.
- On a model requiring EP, published target topology achieves >=70% expert-GEMM utilization for the
  balanced prefill benchmark and a measured decode latency improvement from the selected low-latency
  path; no unsupported claim is made for skewed workloads.
- Any specialized dependency must improve end-to-end goodput >=10% and pass the same failure contract.

### M17 — Speculative decoding

**Depends on:** M12; distributed strategies additionally depend on M14/M15.

**Deliverables**

- `SpeculativeStrategy` contract, scheduler-visible draft/lookahead/verification budgets, KV tentative
  ranges, accept/rollback semantics, and metrics. Implement n-gram first, then one draft-model or
  EAGLE-style strategy only after a separate artifact/placement ADR.
- Verification sampler preserves the target model's distribution and counter-RNG semantics. Normal
  decode remains an immediate fallback.

**Tests**

- Unit: draft tree/ranges, acceptance probabilities, bonus token, tentative KV commit/rollback, stop
  interaction, and budget accounting.
- Integration: continuous/chunked/prefix-cached requests switch strategy and fall back mid-request;
  distributed case follows supported placement.
- Correctness: greedy equivalence and statistical distribution tests against target-only decoding for
  rejection/acceptance edge cases.
- Stress: zero/high acceptance, rapid cancellation, max lookahead, page-boundary rollback, and graph
  bucket churn.
- Failure: draft model/backend failure, invalid token, verifier failure, tentative allocation shortage,
  and unsupported sampling config fall back without corrupting committed KV.
- Performance: acceptance rate, target forward passes saved, draft/verify latency, extra KV memory,
  TPOT and goodput by output/workload.

**Completion criteria**

- Target distribution/equivalence tests pass and committed block tables are identical to normal decode
  after all rollback scenarios.
- Each enabled strategy improves end-to-end TPOT or SLO goodput >=15% on its declared workload after
  accounting for draft compute/memory; otherwise it remains experimental/off by default.

### M18 — Disaggregated prefill/decode and hierarchical KV

**Depends on:** M11, M14, and M15; heterogeneous EP layouts also depend on M16.

**Deliverables**

- Global routing, prefill/decode worker roles, directory/placement service, versioned transfer FSM and
  descriptors, destination reservation, source pin/ack protocol, layout mapper, CUDA P2P transport,
  and optional qualified NIXL backend.
- Same-layout first, then explicitly tested heterogeneous TP/PP mapping. Async overlap, checksums,
  idempotent retries, recompute fallback, transfer metrics/traces, and rolling drain.
- Optional HBM/host/remote tier interface and one host/remote-cache prototype; cost model compares
  transfer/reuse/recompute. NVMe is research-only until data supports it.

**Tests**

- Unit: global/session IDs, descriptor schema, layout mapping, page generations, transfer FSM,
  idempotency, timeout/lease accounting, and tier cost decisions.
- Integration: same-node P/D first; multi-node transport; supported heterogeneous TP/PP; local/remote
  prefix hit; transfer overlap; drain and reroute between requests.
- Correctness: transferred KV tensors/checksums, next-token logits, and full generation match local
  recomputation for all supported layouts/page boundaries and prefix-hit combinations.
- Stress: high concurrent transfers, backpressure, source/destination churn, duplicate/lost control
  messages, remote-cache eviction, and multi-hour network/worker soak.
- Failure: source/destination/rendezvous death, partial-rank transfer, timeout, checksum/fingerprint/
  layout mismatch, stale destination generation, RDMA error, and lost ack either recompute safely or
  fail once without leaks.
- Performance: TTFT/TPOT/goodput versus aggregated serving, KV bytes/bandwidth/latency, transfer/
  compute overlap, pool utilization, and cost by prompt/output distribution and topology.

**Completion criteria**

- No transferred-KV correctness difference from local recompute; source/destination page accounting
  returns exactly after every injected failure and retry.
- For the declared long-prompt interactive workload, P/D improves SLO goodput >=20% or p99 TPOT >=30%
  at no more than 10% TTFT regression versus the best aggregated topology. Otherwise ship the
  architecture as experimental and document the crossover instead of enabling it by default.
- Heterogeneous mappings are advertised only per exact capability matrix; all other pairs reject at
  planning. A 24-hour multi-node soak has no orphan transfer session or unbounded memory growth.

## 20. Major risks, unresolved questions, and required experiments

These questions must be closed by the named milestone. An answer based only on intuition is not a
decision.

| Risk/question | Why it matters | Required experiment/decision point |
|---|---|---|
| Supported CUDA, compiler, and SM baseline | C++23 host support and FlashInfer/CUTLASS/HPC-Ops pins may conflict | M0 build matrix; isolate CUDA translation units; publish exact supported pins |
| Existing tokenizer fidelity and maintenance | Token drift makes every model result wrong; incremental decode/thread safety may be incomplete | M3 10k differential corpus, fuzzing, API/license review; replace adapter if any supported-case mismatch remains |
| Safetensors C++ parser safety | Model files are untrusted input and available C++ readers may have incomplete validation | M0/M3 audit candidate against official Rust behavior/conformance corpus, fuzz all offsets/shapes, and retain only after security gate |
| FlashInfer native C++ integration stability | Upstream is Python/JIT-oriented and CUDA compatibility changes | M4 native minimal integration spike, build-time/startup/perf comparison; retain backend boundary and fallback |
| Beast submodule completeness | Beast depends on other Boost components; current repository declares only Beast | M0 prove hermetic CMake closure or pin required Boost modules/package |
| Folly value versus cost | It can simplify async structures but brings a large dependency closure | M0/M1 implement with C++23/Asio first; retain Folly only for a measured missing capability |
| KV page size and layer packing | Affects fragmentation, attention kernels, metadata, prefix granularity, and transfer | M6 sweep 8/16/32+ tokens and supported layouts on target prompts/GPUs; ADR per backend if needed |
| Reserve full prompt/output versus incremental admission | Full reserve avoids thrash but lowers utilization; incremental reserve can deadlock/preempt heavily | M8 scheduler simulation and near-OOM goodput/fairness tests; choose safe default and watermarks |
| Mixed prefill/decode versus alternating batches | Backend support and interference differ by GPU/workload | M8 compare p99 TPOT/goodput across chunk sizes and phase policies; capability-gate mixed batches |
| Recompute versus host swap preemption | Depends on context, PCIe/NVLink, and memory pressure | Measure after M8; do not implement host swap unless crossover is meaningful |
| Prefix structure: radix edges vs chained page hashes | CPU cost, memory overhead, eviction, and remote lookup differ | M11 prototype both under 100k+ pages and agent/chat prefix distributions |
| Prefix security/isolation | Cross-tenant reuse may leak timing/data and hash collisions can corrupt output | Default tenant-private; M11 collision/namespace audit before any opt-in shared scope |
| CUDA graph bucket policy | Padding raises wasted compute and stable arenas consume memory | M12 derive buckets from traces; require graph hit/memory/goodput gates |
| GPU sampling backend | Third-party/custom kernels may differ in RNG and penalty order | M12 differential and statistical tests before replacing CPU reference |
| Quantization order | Hardware and artifact ecosystem determine value; each format multiplies kernels | M13 choose FP8 plus one INT4 format from real target hardware/models and predeclare quality gates |
| TP collective algorithm/topology | NCCL performance varies sharply across PCIe/NVLink/multi-node | M14 topology sweep; do not advertise arbitrary TP degrees |
| PP scheduling model | Continuous decode feedback and variable microbatches create bubbles/state complexity | M15 simulator plus PP2 prototype; reject topologies with poor measured goodput |
| EP transport backend | NCCL grouped P2P is portable but may be too slow for low-latency decode | M16 compare NCCL with one qualified specialized backend on balanced/skewed routing |
| Local-SPMD placement compiler | Could reduce handwritten parallel code but is a large compiler project | Keep explicit `ParallelPlan`; reconsider only after two architectures expose repeated placement bugs |
| P/D crossover and heterogeneous layout | Transfer and remapping can erase interference gains | M18 measure prompt/output/topology matrix; same-layout first; default off without positive SLO result |
| Distributed failure semantics | NCCL does not provide transparent rank recovery | M14-M18 fail/drain replica first; elastic groups and in-flight migration require a later ADR |
| Remote KV hierarchy | Storage/network may be slower and more complex than recompute | M18 cost-model prototype with observed reuse; NVMe remains research until it wins end to end |
| Target SLO and reference hardware | Absolute performance gates are meaningless without product workload | M0 records hardware; M7 defines initial workload; M10 publishes preview SLO and goodput report |

Features not assigned to milestones—multimodality, adapters/LoRA, beam search, constrained/structured
decoding, context parallelism, sliding-window/recurrent state caches, request migration, autoscaling,
and non-NVIDIA backends—require separate ADRs and roadmaps. The existing boundaries should permit
them, but “permitted” is not a commitment or evidence that the abstractions are sufficient.

## 21. Primary references and applicability notes

- [vLLM architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview/): API/engine/worker
  process responsibilities. InferX adopts the separation of responsibility, not the Python/ZMQ
  implementation.
- [vLLM scheduler source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py):
  scheduler output, KV allocation, preemption, chunking, and connector coordination. InferX keeps the
  plan/resource transaction smaller and typed before adding comparable features.
- [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/): separate
  scheduler- and worker-side connector roles. InferX similarly separates transfer planning from data
  movement, while treating the upstream feature's experimental status as a caution.
- [vLLM Model Runner V2 design](https://docs.vllm.ai/en/latest/design/model_runner_v2/): async-first
  preparation and stable buffer/race concerns. InferX applies these only after its eager path and
  ownership epochs are tested.
- [SGLang paper](https://arxiv.org/abs/2312.07104) and
  [RadixCache source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py):
  prefix-oriented reuse and scheduling. InferX initially caches only immutable full pages with a
  stricter namespace and single-writer metadata.
- [SGLang scheduling policy source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py):
  cache-aware policies and chunk admission. InferX retains deterministic FCFS as its reference policy
  and makes cache-aware selection optional after measurement.
- [SGLang expert parallelism documentation](https://docs.sglang.io/docs/advanced_features/expert_parallelism):
  separate high-throughput prefill and low-latency decode communication modes. InferX uses this as a
  benchmark hypothesis, not as justification to bind core MoE semantics to one communication library.
- [TensorRT-LLM architecture overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/overview.html):
  executor decomposition into scheduler, resource manager, model engine, sampler, and output loop.
- [TensorRT-LLM scheduler documentation](https://nvidia.github.io/TensorRT-LLM/torch/scheduler.html):
  capacity scheduling versus microbatch scheduling. InferX expresses the same useful distinction via
  resource snapshots and transactional reservations.
- [TensorRT-LLM KV cache system](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html):
  preallocated page pools, prefix reuse, and eviction. InferX does not assume its particular
  all-layers-in-a-block layout until M6 experiments.
- [TensorRT-LLM disaggregated serving](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html):
  modular KV exchange, heterogeneous parallel layouts, global request identity, and transfer overlap.
  These inform M18's descriptor, mapper, lease, and acknowledgement design.
- [TokenSpeed repository](https://github.com/lightseekorg/tokenspeed): typed finite-state request/KV
  ownership, C++ control plane, kernel registry, and local-SPMD direction. InferX adopts typed states
  and explicit placement but not its Python execution plane or preview compiler architecture.
- [FlashInfer repository/documentation](https://github.com/flashinfer-ai/flashinfer): candidate
  attention/sampling/MoE kernels and backend selection. It remains a qualified adapter because its
  packaging and supported CUDA versions evolve.
- [NVIDIA CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/index.html): C++ CUDA building
  blocks for specialized GEMM/fusion. cuBLASLt remains the initial production GEMM.
- [Tencent HPC-Ops repository](https://github.com/Tencent/hpc-ops): candidate modern NVIDIA kernels
  and useful comparison implementations. Its narrower hardware and packaging requirements keep it
  experimental until M12/M16 evidence justifies integration.

When an upstream design changes, update the relevant ADR and capability decision; do not let a link
silently redefine InferX behavior. The contracts, tests, and completion gates in this roadmap remain
the authority for this project.
