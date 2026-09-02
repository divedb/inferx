// M5 semantic-layer tests: exact operation order, stable trace names, CPU
// numeric determinism (m5.md sections 8 and 16).

#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/model/forward_batch.h"
#include "inferx/model/intermediate_trace.h"
#include "inferx/model/llama_for_causal_lm.h"
#include "inferx/model/model_spec.h"
#include "inferx/model/model_weights.h"
#include "inferx/ops/cpu_op_executor.h"
#include "inferx/ops/op_executor.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/device.h"
#include "inferx/tensor/dtype.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::model {
namespace {

using ops::OpExecutor;

LlamaSpec MakeSpec() {
  LlamaSpec llama;
  llama.vocab_size = 16;
  llama.hidden_size = 32;
  llama.intermediate_size = 64;
  llama.num_hidden_layers = 1;
  llama.num_attention_heads = 4;
  llama.num_key_value_heads = 2;
  llama.head_dim = 8;
  llama.max_position_embeddings = 64;
  llama.rms_norm_eps = 1e-5;
  llama.rope_theta = 10000.0;
  llama.tie_word_embeddings = false;
  return llama;
}

struct HostBuffer {
  Buffer buffer;
  MutableTensorView view;

  static HostBuffer Make(std::span<const uint64_t> shape, Dtype dtype, uint64_t bytes,
                         MemoryCategory category) {
    CpuAllocator allocator;
    auto buffer = allocator.Allocate(AllocationRequest{Device::Host(), MemoryKind::kHost,
                                                       ByteCount(bytes), ByteCount(64), category});
    auto created = Shape::Create(shape).value();
    auto strides = Strides::Contiguous(created).value();
    auto window = buffer->MutableView(ByteRange{ByteCount(0), ByteCount(bytes)}).value();
    auto view = MutableTensorView::Create(window, dtype, created, strides).value();
    return HostBuffer{std::move(*buffer), view};
  }
};

class RecordingExecutor final : public OpExecutor {
 public:
  absl::Status Embedding(const ops::EmbeddingRequest&) override {
    calls_.push_back("embedding");
    return absl::OkStatus();
  }
  absl::Status Gemm(const ops::GemmRequest&) override {
    calls_.push_back("gemm");
    return absl::OkStatus();
  }
  absl::Status RmsNorm(const ops::RmsNormRequest&) override {
    calls_.push_back("rmsnorm");
    return absl::OkStatus();
  }
  absl::Status Rope(const ops::RopeRequest&) override {
    calls_.push_back("rope");
    return absl::OkStatus();
  }
  absl::Status SwiGlu(const ops::SwiGluRequest&) override {
    calls_.push_back("swiglu");
    return absl::OkStatus();
  }
  absl::Status Residual(const ops::ResidualRequest&) override {
    calls_.push_back("residual");
    return absl::OkStatus();
  }
  absl::Status Attention(const ops::AttentionRequest&) override {
    calls_.push_back("attention");
    return absl::OkStatus();
  }
  absl::Status Logits(const ops::LogitsRequest&) override {
    calls_.push_back("logits");
    return absl::OkStatus();
  }
  absl::Status Copy(const TensorView&, const MutableTensorView&) override {
    calls_.push_back("copy");
    return absl::OkStatus();
  }

  std::vector<std::string> calls_;
};

class RecordingTrace final : public IntermediateTraceSink {
 public:
  void OnIntermediate(std::string_view name, const TensorView&) override {
    names_.emplace_back(name);
  }
  std::vector<std::string> names_;
};

struct SemanticsFixture {
  LlamaSpec llama = MakeSpec();
  ModelSpec spec{llama};
  std::vector<HostBuffer> storage;
  std::vector<MutableTensorView> kv_keys;
  std::vector<MutableTensorView> kv_values;
  ActivationBuffers activations;
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  HostBuffer tokens_tensor;
  HostBuffer positions_tensor;

  SemanticsFixture(uint64_t token_count) {
    const uint64_t hidden = llama.hidden_size;
    const uint64_t query = llama.num_attention_heads * llama.head_dim;
    const uint64_t qkv = llama.num_key_value_heads * llama.head_dim;
    const uint64_t intermediate = llama.intermediate_size;

    auto buffer = [&](std::span<const uint64_t> shape, uint64_t elements) -> MutableTensorView {
      storage.push_back(
          HostBuffer::Make(shape, Dtype::kFloat32, elements * 4, MemoryCategory::kTest));
      return storage.back().view;
    };
    activations.hidden = buffer(std::array<uint64_t, 2>{token_count, hidden}, token_count * hidden);
    activations.normed = buffer(std::array<uint64_t, 2>{token_count, hidden}, token_count * hidden);
    activations.query =
        buffer(std::array<uint64_t, 3>{token_count, llama.num_attention_heads, llama.head_dim},
               token_count * query);
    activations.key =
        buffer(std::array<uint64_t, 3>{token_count, llama.num_key_value_heads, llama.head_dim},
               token_count * qkv);
    activations.value =
        buffer(std::array<uint64_t, 3>{token_count, llama.num_key_value_heads, llama.head_dim},
               token_count * qkv);
    activations.attn_out =
        buffer(std::array<uint64_t, 3>{token_count, llama.num_attention_heads, llama.head_dim},
               token_count * query);
    activations.projected =
        buffer(std::array<uint64_t, 2>{token_count, hidden}, token_count * hidden);
    activations.gate =
        buffer(std::array<uint64_t, 2>{token_count, intermediate}, token_count * intermediate);
    activations.up =
        buffer(std::array<uint64_t, 2>{token_count, intermediate}, token_count * intermediate);
    activations.activated =
        buffer(std::array<uint64_t, 2>{token_count, intermediate}, token_count * intermediate);
    activations.down_out =
        buffer(std::array<uint64_t, 2>{token_count, hidden}, token_count * hidden);
    activations.final_norm =
        buffer(std::array<uint64_t, 2>{token_count, hidden}, token_count * hidden);
    activations.logits = buffer(std::array<uint64_t, 2>{1, llama.vocab_size}, llama.vocab_size);

    for (uint32_t layer = 0; layer < llama.num_hidden_layers; ++layer) {
      const std::array<uint64_t, 4> cache{1, llama.max_position_embeddings,
                                          llama.num_key_value_heads, llama.head_dim};
      const uint64_t elements =
          llama.max_position_embeddings * llama.num_key_value_heads * llama.head_dim;
      storage.push_back(
          HostBuffer::Make(cache, Dtype::kFloat32, elements * 4, MemoryCategory::kTest));
      kv_keys.push_back(storage.back().view);
      storage.push_back(
          HostBuffer::Make(cache, Dtype::kFloat32, elements * 4, MemoryCategory::kTest));
      kv_values.push_back(storage.back().view);
    }

    for (uint64_t index = 0; index < token_count; ++index) {
      tokens.push_back(static_cast<int32_t>((index * 3 + 1) % llama.vocab_size));
      positions.push_back(static_cast<int32_t>(index));
    }
    tokens_tensor = HostBuffer::Make(std::array<uint64_t, 1>{token_count}, Dtype::kInt32,
                                     token_count * 4, MemoryCategory::kTest);
    positions_tensor = HostBuffer::Make(std::array<uint64_t, 1>{token_count}, Dtype::kInt32,
                                        token_count * 4, MemoryCategory::kTest);
    auto token_bytes = tokens_tensor.view.buffer().HostBytes().value();
    auto position_bytes = positions_tensor.view.buffer().HostBytes().value();
    std::memcpy(token_bytes.data(), tokens.data(), tokens.size() * 4);
    std::memcpy(position_bytes.data(), positions.data(), positions.size() * 4);
  }

  // Random-but-deterministic FP32 weights keyed by name.
  std::vector<HostBuffer> weight_storage;
  TensorView Weight(std::span<const uint64_t> shape) {
    uint64_t elements = 1;
    for (const uint64_t dim : shape) elements *= dim;
    weight_storage.push_back(
        HostBuffer::Make(shape, Dtype::kFloat32, elements * 4, MemoryCategory::kTest));
    auto bytes = weight_storage.back().view.buffer().HostBytes().value();
    auto* data = reinterpret_cast<float*>(bytes.data());
    uint32_t state = 0x9E3779B9U + static_cast<uint32_t>(elements);
    for (uint64_t index = 0; index < elements; ++index) {
      state = state * 1664525U + 1013904223U;
      data[index] = static_cast<float>(static_cast<int32_t>(state >> 8) % 200 - 100) / 50.0F;
    }
    return weight_storage.back().view.AsConst();
  }

  ModelWeights MakeWeights() {
    const uint64_t hidden = llama.hidden_size;
    const uint64_t query = llama.num_attention_heads * llama.head_dim;
    const uint64_t qkv = llama.num_key_value_heads * llama.head_dim;
    std::vector<LayerWeights> layers;
    LayerWeights layer;
    layer.input_norm = Weight(std::array<uint64_t, 1>{hidden});
    layer.query = Weight(std::array<uint64_t, 2>{query, hidden});
    layer.key = Weight(std::array<uint64_t, 2>{qkv, hidden});
    layer.value = Weight(std::array<uint64_t, 2>{qkv, hidden});
    layer.attention_output = Weight(std::array<uint64_t, 2>{hidden, query});
    layer.post_norm = Weight(std::array<uint64_t, 1>{hidden});
    layer.mlp_gate = Weight(std::array<uint64_t, 2>{llama.intermediate_size, hidden});
    layer.mlp_up = Weight(std::array<uint64_t, 2>{llama.intermediate_size, hidden});
    layer.mlp_down = Weight(std::array<uint64_t, 2>{hidden, llama.intermediate_size});
    layers.push_back(std::move(layer));
    return ModelWeights::Create(llama, Weight(std::array<uint64_t, 2>{llama.vocab_size, hidden}),
                                Weight(std::array<uint64_t, 1>{hidden}),
                                Weight(std::array<uint64_t, 2>{llama.vocab_size, hidden}),
                                std::move(layers))
        .value();
  }

  ForwardBatch MakeBatch(ActivationBuffers& buffers, IntermediateTraceSink* trace) {
    ForwardBatch batch;
    batch.work = WorkKind::kPrefill;
    batch.num_tokens = tokens.size();
    batch.token_ids = tokens_tensor.view.AsConst();
    batch.positions = positions_tensor.view.AsConst();
    batch.host_token_ids = tokens;
    batch.host_positions = positions;
    batch.key_cache = kv_keys;
    batch.value_cache = kv_values;
    batch.kv_append_begin = 0;
    batch.activations = &buffers;
    batch.trace = trace;
    return batch;
  }
};

TEST(M5SemanticsTest, CreateRejectsIncompleteSpec) {
  LlamaSpec llama;
  ModelSpec spec{llama};
  auto created = LlamaForCausalLM::Create(std::move(spec));
  EXPECT_EQ(created.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(M5SemanticsTest, ExactOperationOrder) {
  SemanticsFixture fixture(3);
  auto semantics = LlamaForCausalLM::Create(fixture.spec).value();
  ModelWeights weights = fixture.MakeWeights();
  RecordingExecutor executor;
  auto batch = fixture.MakeBatch(fixture.activations, nullptr);
  ASSERT_TRUE(semantics.Forward(batch, weights, executor).ok())
      << semantics.Forward(batch, weights, executor);

  const std::vector<std::string> expected = {
      "embedding", "rmsnorm", "gemm",     "gemm",    "gemm",   "rope",
      "attention", "gemm",    "residual", "rmsnorm", "gemm",   "gemm",
      "swiglu",    "gemm",    "residual", "rmsnorm", "logits",
  };
  EXPECT_EQ(executor.calls_, expected);
}

TEST(M5SemanticsTest, StableTraceNames) {
  SemanticsFixture fixture(2);
  auto semantics = LlamaForCausalLM::Create(fixture.spec).value();
  ModelWeights weights = fixture.MakeWeights();
  ops::CpuOpExecutor executor;
  RecordingTrace trace;
  auto batch = fixture.MakeBatch(fixture.activations, &trace);
  ASSERT_TRUE(semantics.Forward(batch, weights, executor).ok())
      << semantics.Forward(batch, weights, executor);

  const std::vector<std::string> expected = {
      "embedding",
      "layers.0.input_norm",
      "layers.0.q_proj",
      "layers.0.k_proj",
      "layers.0.v_proj",
      "layers.0.q_rope",
      "layers.0.k_rope",
      "layers.0.attention",
      "layers.0.o_proj",
      "layers.0.post_attention_residual",
      "layers.0.post_attention_norm",
      "layers.0.mlp_gate",
      "layers.0.mlp_up",
      "layers.0.mlp_activated",
      "layers.0.mlp_down",
      "layers.0.output",
      "final_norm",
      "logits",
  };
  EXPECT_EQ(trace.names_, expected);
}

TEST(M5SemanticsTest, CpuForwardIsDeterministicAndFinite) {
  SemanticsFixture first(5);
  auto semantics = LlamaForCausalLM::Create(first.spec).value();
  ModelWeights weights = first.MakeWeights();
  ops::CpuOpExecutor executor;
  auto batch = first.MakeBatch(first.activations, nullptr);
  ASSERT_TRUE(semantics.Forward(batch, weights, executor).ok())
      << semantics.Forward(batch, weights, executor);
  auto logits_bytes = first.activations.logits.buffer().HostBytes().value();
  const auto* logits = reinterpret_cast<const float*>(logits_bytes.data());

  SemanticsFixture second(5);
  auto batch_again = second.MakeBatch(second.activations, nullptr);
  ASSERT_TRUE(semantics.Forward(batch_again, weights, executor).ok());
  auto other_bytes = second.activations.logits.buffer().HostBytes().value();
  const auto* other = reinterpret_cast<const float*>(other_bytes.data());

  for (uint64_t id = 0; id < first.llama.vocab_size; ++id) {
    EXPECT_TRUE(std::isfinite(logits[id])) << id;
    EXPECT_EQ(logits[id], other[id]) << id;
  }
}

TEST(M5SemanticsTest, DecodeCarriesExactlyOneToken) {
  SemanticsFixture fixture(2);
  auto semantics = LlamaForCausalLM::Create(fixture.spec).value();
  ModelWeights weights = fixture.MakeWeights();
  RecordingExecutor executor;
  auto batch = fixture.MakeBatch(fixture.activations, nullptr);
  batch.work = WorkKind::kDecode;
  EXPECT_EQ(semantics.Forward(batch, weights, executor).code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx::model
