// M5 runtime-contract tests: deterministic memory planning and the
// contiguous KV transaction (m5.md sections 9 and 11).

#include <array>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/artifacts/artifact_tensor.h"
#include "inferx/base/id.h"
#include "inferx/model/model_spec.h"
#include "inferx/model/weight_plan.h"
#include "inferx/runtime/contiguous_kv_cache.h"
#include "inferx/runtime/model_memory_plan.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/device.h"
#include "inferx/tensor/dtype.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::runtime {
namespace {

model::ModelSpec MakeModelSpec() {
  model::LlamaSpec llama;
  llama.vocab_size = 4;
  llama.hidden_size = 2;
  llama.intermediate_size = 4;
  llama.num_hidden_layers = 1;
  llama.num_attention_heads = 1;
  llama.num_key_value_heads = 1;
  llama.head_dim = 2;
  llama.max_position_embeddings = 16;
  llama.rms_norm_eps = 1e-5;
  llama.rope_theta = 10000.0;
  llama.tie_word_embeddings = false;
  return model::ModelSpec{llama};
}

TEST(M5MemoryPlanTest, ComputesDeterministicGeometry) {
  model::ModelSpec spec = MakeModelSpec();
  MemoryPlanRequest request;
  request.spec = &spec;
  request.weight_plan = nullptr;
  request.max_prefill_tokens = 8;
  request.context_capacity = 12;

  auto plan = ModelMemoryPlanner::Plan(request);
  ASSERT_FALSE(plan.ok());  // catalog/weight plan are required inputs
  EXPECT_EQ(plan.status().code(), absl::StatusCode::kInvalidArgument);

  // Weight plan over one parameter plus one alias, matching the catalog the
  // planner is fed through the request.
  std::vector<model::WeightPlanItem> items;
  model::TensorSource source{artifacts::SafeRelativePath::Parse("model.safetensors").value(),
                             artifacts::FileIdentity{},
                             artifacts::ArtifactByteRange{},
                             artifacts::ArtifactDtype::kF16,
                             {4, 2},
                             artifacts::Digest256{}};
  items.push_back(model::WeightPlanItem{model::ParameterId(0), source, model::TransformSpec{},
                                        model::ShardKind::kReplicatedFullTensor});
  model::WeightPlan weight_plan;
  weight_plan.items = items;
  weight_plan.aliases.push_back(
      model::ParameterAlias{model::ParameterId(1), model::ParameterId(0)});

  request.weight_plan = &weight_plan;
  plan = ModelMemoryPlanner::Plan(request);
  ASSERT_TRUE(plan.ok()) << plan.status();

  // One F16 [4,2] tensor materializes as 8 FP32 bytes at the 256-byte floor.
  ASSERT_EQ(plan->weights.size(), 2U);
  EXPECT_EQ(plan->weights[0].offset, 0U);
  EXPECT_EQ(plan->weights[0].bytes, 32U);  // 8 elements materialized as FP32
  EXPECT_EQ(plan->weights[0].alignment, 256U);
  EXPECT_TRUE(plan->weights[1].alias);
  EXPECT_EQ(plan->weights[1].bytes, 0U);
  // One 8-byte placement starting at the aligned floor: the arena ends at
  // the item end with no trailing padding.
  EXPECT_EQ(plan->weight_arena_bytes, 32U);

  // KV = 2 * L * C * Nkv * D * 4 with the fixed geometry.
  EXPECT_EQ(plan->kv_bytes, 2U * 1U * 12U * 1U * 2U * 4U);
  // Final-row logits: vocab * 4.
  EXPECT_EQ(plan->logits_bytes, 4U * 4U);

  // Determinism: an identical request yields an identical plan.
  auto again = ModelMemoryPlanner::Plan(request);
  ASSERT_TRUE(again.ok());
  EXPECT_EQ(plan->total_device_bytes, again->total_device_bytes);
  EXPECT_EQ(plan->weights[0].offset, again->weights[0].offset);
}

TEST(M5MemoryPlanTest, RejectsImpossibleEnvelopes) {
  model::ModelSpec spec = MakeModelSpec();
  MemoryPlanRequest request;
  request.spec = &spec;
  request.max_prefill_tokens = 0;
  auto plan = ModelMemoryPlanner::Plan(request);
  EXPECT_EQ(plan.status().code(), absl::StatusCode::kInvalidArgument);

  request.max_prefill_tokens = 8;
  request.context_capacity = 100;  // above the model maximum (16)
  plan = ModelMemoryPlanner::Plan(request);
  EXPECT_EQ(plan.status().code(), absl::StatusCode::kInvalidArgument);
}

struct KvFixture {
  std::vector<Buffer> buffers;
  std::vector<MutableTensorView> keys;
  std::vector<MutableTensorView> values;

  KvFixture() {
    CpuAllocator allocator;
    const std::array<uint64_t, 4> shape{1, 8, 1, 2};
    const uint64_t bytes = 8 * 1 * 2 * 4;
    for (int side = 0; side < 2; ++side) {
      buffers.push_back(
          allocator
              .Allocate(AllocationRequest{Device::Host(), MemoryKind::kHost, ByteCount(bytes),
                                          ByteCount(64), MemoryCategory::kTest})
              .value());
      auto created = Shape::Create(shape).value();
      auto strides = Strides::Contiguous(created).value();
      auto window = buffers.back().MutableView(ByteRange{ByteCount(0), ByteCount(bytes)}).value();
      auto view = MutableTensorView::Create(window, Dtype::kFloat32, created, strides).value();
      (side == 0 ? keys : values).push_back(view);
    }
  }
};

TEST(M5ContiguousKvTest, TransactionLifecycle) {
  KvFixture fixture;
  auto cache = ContiguousKvCache::Create(1, 8, fixture.keys, fixture.values).value();
  EXPECT_EQ(cache.state(), ContiguousKvCache::State::kEmpty);

  EXPECT_TRUE(cache.Acquire(SequenceId(7)).ok());
  EXPECT_EQ(cache.state(), ContiguousKvCache::State::kActive);
  EXPECT_EQ(cache.committed_length(), 0U);

  // Append must begin exactly at the committed boundary.
  auto mismatched = cache.PrepareAppend(3, 2, ExecutionTicketId(1));
  EXPECT_EQ(mismatched.status().code(), absl::StatusCode::kFailedPrecondition);

  auto append = cache.PrepareAppend(0, 3, ExecutionTicketId(1));
  ASSERT_TRUE(append.ok());
  EXPECT_EQ(cache.state(), ContiguousKvCache::State::kAppendPending);
  append->Commit();
  EXPECT_EQ(cache.committed_length(), 3U);
  EXPECT_EQ(cache.state(), ContiguousKvCache::State::kActive);

  // Rollback leaves the length unchanged.
  auto rolled = cache.PrepareAppend(3, 2, ExecutionTicketId(2));
  ASSERT_TRUE(rolled.ok());
  rolled->Rollback();
  EXPECT_EQ(cache.committed_length(), 3U);

  // Out-of-range append.
  auto beyond = cache.PrepareAppend(3, 6, ExecutionTicketId(3));
  EXPECT_EQ(beyond.status().code(), absl::StatusCode::kOutOfRange);

  // Reset bumps the generation and returns to the empty (re-acquirable)
  // state without a pending transaction.
  EXPECT_TRUE(cache.Reset().ok());
  EXPECT_EQ(cache.committed_length(), 0U);
  EXPECT_EQ(cache.generation(), KvSequenceGeneration(1));
  EXPECT_EQ(cache.state(), ContiguousKvCache::State::kEmpty);
  EXPECT_TRUE(cache.Acquire(SequenceId(8)).ok());
}

TEST(M5ContiguousKvTest, DestructorRollsBackPendingAppend) {
  KvFixture fixture;
  auto cache = ContiguousKvCache::Create(1, 8, fixture.keys, fixture.values).value();
  ASSERT_TRUE(cache.Acquire(SequenceId(1)).ok());
  {
    auto append = cache.PrepareAppend(0, 2, ExecutionTicketId(1));
    ASSERT_TRUE(append.ok());
  }  // destroyed without Commit: rolls back
  EXPECT_EQ(cache.state(), ContiguousKvCache::State::kActive);
  EXPECT_EQ(cache.committed_length(), 0U);
  auto again = cache.PrepareAppend(0, 1, ExecutionTicketId(2));
  EXPECT_TRUE(again.ok());
}

TEST(M5ContiguousKvTest, RejectsBadViewGeometry) {
  KvFixture fixture;
  CpuAllocator allocator;
  const std::array<uint64_t, 4> wrong{1, 4, 1, 2};
  const uint64_t bytes = 4 * 1 * 2 * 4;
  auto buffer = allocator
                    .Allocate(AllocationRequest{Device::Host(), MemoryKind::kHost, ByteCount(bytes),
                                                ByteCount(64), MemoryCategory::kTest})
                    .value();
  auto created = Shape::Create(wrong).value();
  auto strides = Strides::Contiguous(created).value();
  auto window = buffer.MutableView(ByteRange{ByteCount(0), ByteCount(bytes)}).value();
  auto view = MutableTensorView::Create(window, Dtype::kFloat32, created, strides).value();
  std::vector<MutableTensorView> keys{view};
  auto cache = ContiguousKvCache::Create(1, 8, keys, fixture.values);
  EXPECT_EQ(cache.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx::runtime
