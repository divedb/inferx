#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/cpu_reference_executor.h"
#include "inferx/ops/embedding.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/kernel_registry.h"
#include "inferx/ops/logits.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/ops/rope.h"
#include "inferx/ops/swiglu.h"
#include "inferx/ops/warmup.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::ops {
namespace {

struct OwnedTensor {
  Buffer buffer;
  MutableTensorView mutable_view;
  TensorView view;
};

template <typename T>
OwnedTensor MakeTensor(std::span<const uint64_t> dimensions, DType dtype,
                       std::span<const T> values) {
  auto shape = Shape::Create(dimensions);
  EXPECT_TRUE(shape.ok()) << shape.status();
  auto strides = Strides::Contiguous(*shape);
  EXPECT_TRUE(strides.ok()) << strides.status();
  auto bytes = shape->Bytes(dtype);
  EXPECT_TRUE(bytes.ok()) << bytes.status();
  CpuAllocator allocator;
  auto buffer = allocator.Allocate(AllocationRequest{Device::Host(), MemoryKind::kHost, *bytes,
                                                     ByteCount(alignof(T)), MemoryCategory::kTest});
  EXPECT_TRUE(buffer.ok()) << buffer.status();
  auto buffer_view = buffer->MutableView(ByteRange{ByteCount(0), *bytes});
  EXPECT_TRUE(buffer_view.ok()) << buffer_view.status();
  auto host = buffer_view->HostBytes();
  EXPECT_TRUE(host.ok()) << host.status();
  EXPECT_EQ(host->size(), values.size_bytes());
  if (!values.empty()) std::memcpy(host->data(), values.data(), values.size_bytes());
  auto mutable_view = MutableTensorView::Create(*buffer_view, dtype, *shape, *strides);
  EXPECT_TRUE(mutable_view.ok()) << mutable_view.status();
  return OwnedTensor{std::move(*buffer), *mutable_view, mutable_view->AsConst()};
}

template <typename T, size_t Size>
OwnedTensor MakeTensor(std::span<const uint64_t> dimensions, DType dtype,
                       const std::array<T, Size>& values) {
  return MakeTensor<T>(dimensions, dtype, std::span<const T>(values));
}

std::span<const float> Values(const OwnedTensor& tensor) {
  auto bytes = tensor.view.buffer().HostBytes();
  EXPECT_TRUE(bytes.ok()) << bytes.status();
  return std::span<const float>(reinterpret_cast<const float*>(bytes->data()),
                                bytes->size() / sizeof(float));
}

void ExpectValues(std::span<const float> actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    EXPECT_FLOAT_EQ(actual[index], expected[index]) << "element " << index;
  }
}

template <size_t Size>
void ExpectValues(std::span<const float> actual, const std::array<float, Size>& expected) {
  ExpectValues(actual, std::span<const float>(expected));
}

TEST(M4ReferenceTest, EmbeddingValidatesEveryIdBeforeWriting) {
  constexpr std::array<uint64_t, 1> kIdShape{2};
  constexpr std::array<uint64_t, 2> kWeightShape{3, 2};
  constexpr std::array<uint64_t, 2> kOutputShape{2, 2};
  auto ids = MakeTensor(kIdShape, DType::kInt32, std::array<int32_t, 2>{1, 3});
  auto weight = MakeTensor(kWeightShape, DType::kFloat32, std::array<float, 6>{1, 2, 3, 4, 5, 6});
  auto output = MakeTensor(kOutputShape, DType::kFloat32, std::array<float, 4>{9, 9, 9, 9});
  absl::Status status = ReferenceEmbedding({ids.view, weight.view, output.mutable_view});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ExpectValues(Values(output), std::array<float, 4>{9, 9, 9, 9});
}

TEST(M4ReferenceTest, GemmUsesOutInWeightLayout) {
  constexpr std::array<uint64_t, 2> kShape{2, 2};
  auto input = MakeTensor(kShape, DType::kFloat32, std::array<float, 4>{1, 2, 3, 4});
  auto weight = MakeTensor(kShape, DType::kFloat32, std::array<float, 4>{5, 6, 7, 8});
  auto output = MakeTensor(kShape, DType::kFloat32, std::array<float, 4>{0, 0, 0, 0});
  ASSERT_TRUE(ReferenceGemm({input.view, weight.view, std::nullopt, output.mutable_view}).ok());
  ExpectValues(Values(output), std::array<float, 4>{17, 23, 39, 53});
}

TEST(M4ReferenceTest, RmsNormAndSwiGluMatchHandValues) {
  constexpr std::array<uint64_t, 2> kShape{1, 2};
  constexpr std::array<uint64_t, 1> kWeightShape{2};
  auto input = MakeTensor(kShape, DType::kFloat32, std::array<float, 2>{3, 4});
  auto weight = MakeTensor(kWeightShape, DType::kFloat32, std::array<float, 2>{1, 2});
  auto normalized = MakeTensor(kShape, DType::kFloat32, std::array<float, 2>{0, 0});
  ASSERT_TRUE(ReferenceRmsNorm({input.view, weight.view, normalized.mutable_view, 1.0e-5F}).ok());
  EXPECT_NEAR(Values(normalized)[0], 3.0F / std::sqrt(12.50001F), 1.0e-6F);
  EXPECT_NEAR(Values(normalized)[1], 8.0F / std::sqrt(12.50001F), 1.0e-6F);

  auto up = MakeTensor(kShape, DType::kFloat32, std::array<float, 2>{2, 3});
  auto activated = MakeTensor(kShape, DType::kFloat32, std::array<float, 2>{0, 0});
  ASSERT_TRUE(ReferenceSwiGlu({input.view, up.view, activated.mutable_view}).ok());
  EXPECT_NEAR(Values(activated)[0], (3.0F / (1.0F + std::exp(-3.0F))) * 2.0F, 1.0e-6F);
}

TEST(M4ReferenceTest, RopeUsesHalfSplitPairing) {
  constexpr std::array<uint64_t, 3> kShape{1, 1, 4};
  constexpr std::array<uint64_t, 1> kPositionShape{1};
  auto query = MakeTensor(kShape, DType::kFloat32, std::array<float, 4>{1, 2, 3, 4});
  auto key = MakeTensor(kShape, DType::kFloat32, std::array<float, 4>{5, 6, 7, 8});
  auto positions = MakeTensor(kPositionShape, DType::kInt32, std::array<int32_t, 1>{0});
  auto query_output = MakeTensor(kShape, DType::kFloat32, std::array<float, 4>{0, 0, 0, 0});
  auto key_output = MakeTensor(kShape, DType::kFloat32, std::array<float, 4>{0, 0, 0, 0});
  ASSERT_TRUE(ReferenceRope({query.view, key.view, positions.view, query_output.mutable_view,
                             key_output.mutable_view, 10'000.0F, 128})
                  .ok());
  ExpectValues(Values(query_output), Values(query));
  ExpectValues(Values(key_output), Values(key));
}

TEST(M4ReferenceTest, AttentionAppendsKvAndMapsGqaHeads) {
  constexpr std::array<uint64_t, 3> kQueryShape{2, 2, 2};
  constexpr std::array<uint64_t, 3> kKvShape{2, 1, 2};
  constexpr std::array<uint64_t, 4> kCacheShape{1, 4, 1, 2};
  auto query =
      MakeTensor(kQueryShape, DType::kFloat32, std::array<float, 8>{0, 0, 0, 0, 0, 0, 0, 0});
  auto key = MakeTensor(kKvShape, DType::kFloat32, std::array<float, 4>{1, 0, 0, 1});
  auto value = MakeTensor(kKvShape, DType::kFloat32, std::array<float, 4>{2, 4, 6, 8});
  auto key_cache =
      MakeTensor(kCacheShape, DType::kFloat32, std::array<float, 8>{0, 0, 0, 0, 0, 0, 0, 0});
  auto value_cache =
      MakeTensor(kCacheShape, DType::kFloat32, std::array<float, 8>{0, 0, 0, 0, 0, 0, 0, 0});
  auto output =
      MakeTensor(kQueryShape, DType::kFloat32, std::array<float, 8>{0, 0, 0, 0, 0, 0, 0, 0});
  constexpr std::array<int32_t, 2> kIndptr{0, 2};
  constexpr std::array<int32_t, 2> kPositions{0, 1};
  constexpr std::array<int32_t, 1> kBefore{0};
  std::array<float, 4> scratch{};
  AttentionRequest request{query.view,
                           key.view,
                           value.view,
                           kIndptr,
                           kIndptr,
                           kPositions,
                           kBefore,
                           key_cache.mutable_view,
                           value_cache.mutable_view,
                           output.mutable_view,
                           ExecutionPhase::kPrefill};
  ASSERT_TRUE(ReferenceAttention(request, scratch).ok());
  ExpectValues(Values(output), std::array<float, 8>{2, 4, 2, 4, 4, 6, 4, 6});
  ExpectValues(Values(key_cache).subspan(0, 4), Values(key));
}

BackendCapability Capability(std::string id, uint32_t priority) {
  BackendCapability capability;
  capability.backend = BackendId::kReference;
  capability.capability_id = std::move(id);
  capability.priority = priority;
  capability.op = OpKind::kGemm;
  capability.device_kind = DeviceKind::kHost;
  capability.input_dtype_mask = DTypeMask(DType::kFloat32);
  capability.weight_dtype_mask = DTypeMask(DType::kFloat32);
  capability.output_dtype_mask = DTypeMask(DType::kFloat32);
  capability.rank = 3;
  capability.dimensions[0] = {0, 8};
  capability.dimensions[1] = {1, 8};
  capability.dimensions[2] = {1, 8};
  capability.minimum_operand_alignment = 4;
  return capability;
}

TEST(M4RegistryTest, SelectionIsIndependentOfRegistrationOrder) {
  KernelKey key;
  key.op = OpKind::kGemm;
  key.rank = 3;
  key.dimensions[0] = 1;
  key.dimensions[1] = 2;
  key.dimensions[2] = 3;
  key.alignment_class = 4;
  KernelRegistry registry(2);
  ASSERT_TRUE(registry.Register(Capability("z", 1)).ok());
  ASSERT_TRUE(registry.Register(Capability("a", 1)).ok());
  ASSERT_TRUE(registry.Freeze().ok());
  auto selection = registry.Select(key);
  ASSERT_TRUE(selection.ok()) << selection.status();
  EXPECT_EQ(selection->capability_id, "a");
  EXPECT_EQ(SerializeKernelKey(key).size(), 110U);
  EXPECT_EQ(StableKernelKeyHash(key), 0xccf6cd57bd9e2f9dULL);
}

TEST(M4RegistryTest, RequiredReferenceSetWarmsTransactionally) {
  LlamaOperatorSpec llama;
  llama.vocab_size = 32;
  llama.hidden_size = 8;
  llama.intermediate_size = 16;
  llama.num_attention_heads = 4;
  llama.num_key_value_heads = 2;
  llama.head_dim = 2;
  OperatorEnvelope envelope;
  envelope.maximum_batch = 2;
  envelope.maximum_prompt_tokens = 4;
  envelope.maximum_context = 8;
  envelope.workspace_limit_bytes = 4096;
  envelope.token_buckets = {1, 4};

  auto keys = BuildRequiredKernelSet(llama, envelope);
  ASSERT_TRUE(keys.ok()) << keys.status();
  ASSERT_FALSE(keys->empty());
  KernelRegistry registry(16);
  ASSERT_TRUE(RegisterReferenceCapabilities(registry).ok());
  auto cache = WarmupKernelSet(
      registry, *keys, keys->size(),
      [](const KernelSelection& selection,
         const KernelKey& key) -> absl::StatusOr<PreparedKernelPlan> {
        return PreparedKernelPlan{key, selection.backend, selection.capability_id,
                                  selection.match.workspace_bytes,
                                  selection.match.workspace_alignment};
      },
      BackendId::kReference);
  ASSERT_TRUE(cache.ok()) << cache.status();
  EXPECT_TRUE(cache->frozen());
  EXPECT_EQ(cache->size(), keys->size());
  for (const KernelKey& key : *keys) EXPECT_NE(cache->Find(key), nullptr);
}

TEST(M4ContractTest, ZeroTokenOperationsValidateAsNoOps) {
  constexpr std::array<uint64_t, 1> kNoTokens{0};
  constexpr std::array<uint64_t, 2> kEmbeddingWeightShape{2, 3};
  constexpr std::array<uint64_t, 2> kEmptyHiddenShape{0, 3};
  auto ids = MakeTensor<int32_t>(kNoTokens, DType::kInt32, std::span<const int32_t>());
  auto embedding_weight = MakeTensor(kEmbeddingWeightShape, DType::kFloat32,
                                     std::array<float, 6>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  auto empty_hidden =
      MakeTensor<float>(kEmptyHiddenShape, DType::kFloat32, std::span<const float>());
  EXPECT_TRUE(
      ReferenceEmbedding({ids.view, embedding_weight.view, empty_hidden.mutable_view}).ok());

  constexpr std::array<uint64_t, 2> kGemmWeightShape{2, 3};
  constexpr std::array<uint64_t, 2> kEmptyGemmShape{0, 2};
  auto gemm_weight = MakeTensor(kGemmWeightShape, DType::kFloat32,
                                std::array<float, 6>{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F});
  auto empty_gemm = MakeTensor<float>(kEmptyGemmShape, DType::kFloat32, std::span<const float>());
  EXPECT_TRUE(
      ReferenceGemm({empty_hidden.view, gemm_weight.view, std::nullopt, empty_gemm.mutable_view})
          .ok());

  constexpr std::array<uint64_t, 1> kNormWeightShape{3};
  auto norm_weight =
      MakeTensor(kNormWeightShape, DType::kFloat32, std::array<float, 3>{1.0F, 1.0F, 1.0F});
  auto empty_norm = MakeTensor<float>(kEmptyHiddenShape, DType::kFloat32, std::span<const float>());
  EXPECT_TRUE(
      ReferenceRmsNorm({empty_hidden.view, norm_weight.view, empty_norm.mutable_view, 1.0e-5F})
          .ok());
}

TEST(M4ContractTest, RejectsPartialAndReadOnlyOperandAliases) {
  constexpr std::array<uint64_t, 2> kMatrixShape{2, 2};
  auto shared =
      MakeTensor(kMatrixShape, DType::kFloat32, std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F});
  auto output =
      MakeTensor(kMatrixShape, DType::kFloat32, std::array<float, 4>{9.0F, 9.0F, 9.0F, 9.0F});
  EXPECT_EQ(ValidateGemm({shared.view, shared.view, std::nullopt, output.mutable_view}).code(),
            absl::StatusCode::kInvalidArgument);
  ExpectValues(Values(output), std::array<float, 4>{9.0F, 9.0F, 9.0F, 9.0F});

  constexpr std::array<uint64_t, 1> kBackingShape{3};
  constexpr std::array<uint64_t, 1> kVectorShape{2};
  auto backing = MakeTensor(kBackingShape, DType::kFloat32, std::array<float, 3>{1.0F, 2.0F, 3.0F});
  auto backing_view = backing.buffer.MutableView(ByteRange{ByteCount(0), backing.buffer.size()});
  ASSERT_TRUE(backing_view.ok()) << backing_view.status();
  auto vector_shape = Shape::Create(kVectorShape);
  ASSERT_TRUE(vector_shape.ok()) << vector_shape.status();
  auto vector_strides = Strides::Contiguous(*vector_shape);
  ASSERT_TRUE(vector_strides.ok()) << vector_strides.status();
  auto left =
      TensorView::Create(backing_view->AsConst(), DType::kFloat32, *vector_shape, *vector_strides);
  ASSERT_TRUE(left.ok()) << left.status();
  auto partial_output = MutableTensorView::Create(*backing_view, DType::kFloat32, *vector_shape,
                                                  *vector_strides, ByteCount(sizeof(float)));
  ASSERT_TRUE(partial_output.ok()) << partial_output.status();
  auto right = MakeTensor(kVectorShape, DType::kFloat32, std::array<float, 2>{4.0F, 5.0F});
  EXPECT_EQ(ValidateResidual({*left, right.view, *partial_output}).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(M4ReferenceTest, ResourceLimitsRejectBeforeMutation) {
  constexpr std::array<uint64_t, 2> kShape{2, 2};
  auto input = MakeTensor(kShape, DType::kFloat32, std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F});
  auto weight = MakeTensor(kShape, DType::kFloat32, std::array<float, 4>{5.0F, 6.0F, 7.0F, 8.0F});
  auto output = MakeTensor(kShape, DType::kFloat32, std::array<float, 4>{9.0F, 9.0F, 9.0F, 9.0F});
  CpuReferenceExecutor executor(ReferenceLimits{3, std::numeric_limits<uint64_t>::max()});
  const absl::Status status =
      executor.Execute({input.view, weight.view, std::nullopt, output.mutable_view});
  EXPECT_EQ(status.code(), absl::StatusCode::kResourceExhausted);
  ExpectValues(Values(output), std::array<float, 4>{9.0F, 9.0F, 9.0F, 9.0F});
}

TEST(M4ReferenceTest, RandomizedGemmMatchesFp64Traversal) {
  std::mt19937 generator(0x4d345245U);
  std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
  constexpr std::array<uint64_t, 2> kInputShape{7, 5};
  constexpr std::array<uint64_t, 2> kWeightShape{3, 5};
  constexpr std::array<uint64_t, 2> kOutputShape{7, 3};
  std::array<float, 35> input_values{};
  std::array<float, 15> weight_values{};
  for (float& value : input_values) value = distribution(generator);
  for (float& value : weight_values) value = distribution(generator);
  auto input = MakeTensor(kInputShape, DType::kFloat32, input_values);
  auto weight = MakeTensor(kWeightShape, DType::kFloat32, weight_values);
  auto output = MakeTensor(kOutputShape, DType::kFloat32, std::array<float, 21>{});
  ASSERT_TRUE(ReferenceGemm({input.view, weight.view, std::nullopt, output.mutable_view}).ok());
  for (size_t row = 0; row < 7; ++row) {
    for (size_t column = 0; column < 3; ++column) {
      double expected = 0.0;
      for (size_t inner = 0; inner < 5; ++inner) {
        expected += static_cast<double>(input_values[row * 5 + inner]) *
                    static_cast<double>(weight_values[column * 5 + inner]);
      }
      EXPECT_NEAR(Values(output)[row * 3 + column], static_cast<float>(expected), 2.0e-6F);
    }
  }
}

TEST(M4RegistryTest, RejectsMalformedKeysAndBoundsPreparedCache) {
  KernelKey invalid_cuda;
  invalid_cuda.device_kind = DeviceKind::kCuda;
  EXPECT_EQ(ValidateKernelKey(invalid_cuda).code(), absl::StatusCode::kInvalidArgument);

  KernelKey invalid_attention;
  invalid_attention.op = OpKind::kAttention;
  invalid_attention.phase = ExecutionPhase::kPrefill;
  invalid_attention.rank = 4;
  invalid_attention.dimensions[0] = 1;
  invalid_attention.dimensions[1] = 4;
  invalid_attention.dimensions[2] = 8;
  invalid_attention.dimensions[3] = 4;
  invalid_attention.query_heads = 4;
  invalid_attention.kv_heads = 2;
  invalid_attention.head_dimension = 4;
  invalid_attention.sequence_bucket = 1;
  invalid_attention.causal = true;
  EXPECT_EQ(ValidateKernelKey(invalid_attention).code(), absl::StatusCode::kInvalidArgument);

  KernelKey key;
  key.op = OpKind::kGemm;
  key.rank = 3;
  key.dimensions[0] = 1;
  key.dimensions[1] = 2;
  key.dimensions[2] = 3;
  PreparedKernelCache cache(1);
  EXPECT_TRUE(cache.Insert({key, BackendId::kReference, "reference.gemm", 0, 1}).ok());
  KernelKey second = key;
  second.dimensions[0] = 2;
  EXPECT_EQ(cache.Insert({second, BackendId::kReference, "reference.gemm", 0, 1}).code(),
            absl::StatusCode::kResourceExhausted);
  EXPECT_TRUE(cache.Freeze().ok());
  EXPECT_EQ(cache.Insert({second, BackendId::kReference, "reference.gemm", 0, 1}).code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(M4RegistryTest, WarmupFailureDoesNotPublishPartialCache) {
  KernelKey first;
  first.op = OpKind::kGemm;
  first.rank = 3;
  first.dimensions[0] = 1;
  first.dimensions[1] = 2;
  first.dimensions[2] = 3;
  first.alignment_class = 4;
  KernelKey second = first;
  second.dimensions[0] = 2;
  const std::array<KernelKey, 2> keys{first, second};
  KernelRegistry registry(2);
  ASSERT_TRUE(registry.Register(Capability("only", 1)).ok());
  size_t prepared = 0;
  auto cache =
      WarmupKernelSet(registry, keys, keys.size(),
                      [&prepared](const KernelSelection& selection,
                                  const KernelKey& key) -> absl::StatusOr<PreparedKernelPlan> {
                        ++prepared;
                        if (prepared == 2)
                          return absl::InternalError("fake.prepare: injected failure");
                        return PreparedKernelPlan{key, selection.backend, selection.capability_id,
                                                  selection.match.workspace_bytes,
                                                  selection.match.workspace_alignment};
                      });
  EXPECT_EQ(cache.status().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(prepared, 2U);
  EXPECT_TRUE(registry.frozen());
}

}  // namespace
}  // namespace inferx::ops
