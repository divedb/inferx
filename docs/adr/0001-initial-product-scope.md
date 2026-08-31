# ADR 0001: Initial product scope

- Status: Proposed
- Date: 2026-08-30
- Deciding owner: repository owner (implementation proceeds provisionally per
  `docs/adr/README.md`)
- Seed: `docs/milestones/m0.md` section 4

## Context

InferX starts from an empty repository (no build system, sources, or CI) and a roadmap
(`docs/plan.md`) that fixes a first vertical slice: load one dense Llama model, tokenize
one prompt, prefill and decode on one NVIDIA GPU, and stream tokens through an embeddable
C++ API. Everything before M5 exists to make that slice reliable. Scope creep in platform,
model coverage, and API surface is the primary schedule risk because each addition widens
the qualification, testing, and packaging matrix before any end-to-end behavior exists.

## Decision

1. **Platform:** Linux x86-64 host, NVIDIA CUDA GPU required for inference behavior;
   CPU builds exist for development, tests, and control-plane code.
2. **Language:** C++23 for host code. CUDA C++ is confined to `platform/cuda` and
   `kernels/` (per ADR 0005 for the language level inside those translation units).
   No Python in the runtime; Python is allowed for developer tooling and reference
   tests only.
3. **First model family:** dense decoder-only Llama architecture, FP16/BF16 weights.
   Qwen-dense, quantization, and MoE come later in the roadmap order.
4. **Artifacts:** Hugging Face-style `config.json`, `tokenizer.json`, safetensors
   shards plus shard index. Token-ID-first API; text API layers on top. Never load
   pickle in the engine.
5. **API:** embeddable asynchronous C++ API first (token IDs in, ordered events out);
   HTTP/OpenAI compatibility arrives in M9 as a server layer above the engine, not
   inside it.
6. **Deployment:** one process, one GPU for the first release; multi-GPU and
   multi-node follow the roadmap's parallelism milestones.

## Alternatives

- **Multi-platform from the start (Linux+macOS+Windows, ROCm):** rejected for M0–M5;
   each lane multiplies CUDA/toolchain qualification with no contribution to the first
   slice. The CMake structure keeps the door open.
- **C++20 host standard:** rejected; C++23 facilities the plan relies on
   (`std::expected`, `std::flat_map` candidates, `std::stop_token` refinements) are
   available on the accepted compiler floor, and choosing 23 now avoids a later
   project-wide flag day.
- **MoE or multimodal first:** rejected; dense Llama is the smallest architecture that
   still exercises attention, KV cache, and sampling.
- **HTTP-first product:** rejected; it couples engine lifecycles to protocol concerns
   before the engine exists.

## Consequences

- Unsupported requests (Windows builds, CPU inference, other model families) are
  answered with "not in initial scope" rather than partial code.
- The dependency set stays minimal until the slice lands (ADR 0003).
- The public C++ API is treated as unstable until M7; no ABI promise is made
  (ADR 0004).
- Feature milestones M1+ inherit a fixed platform assumption and can spend their
  budget on behavior instead of portability.

## Validation evidence

- `cmake --preset dev-clang && cmake --build --preset dev-clang && ctest --preset
  dev-clang` compiles the C++23 sentinel (`tests/compile/cxx23_smoke.cc`) proving the
  language choice on the accepted floor.
- `docs/supported-platforms.md` records which platforms are required/supported/
  experimental/unsupported, seeded from this ADR.

## Supersession

Superseded only by a new ADR linked from here that changes platform, language, first
model family, artifact formats, or API shape; all dependent milestones re-derive their
plans from the successor.
