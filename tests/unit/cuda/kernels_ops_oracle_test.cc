// GPU oracle tests for the TokenSpeed-gap operators (ADR 0032 addendum):
// each new op executes through the unified dispatch surface and is compared
// against the kernels-layer CPU references on the SM89 device.
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "cuda_kernel_backend.h"
#include "gtest/gtest.h"
#include "inferx/kernels/cuda/buffer_access.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/ops/activation.h"
#include "inferx/kernels/ops/layernorm.h"
#include "inferx/kernels/ops/model_fused.h"
#include "inferx/kernels/ops/sampling.h"
#include "inferx/kernels/ops/transform.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"
#include "inferx/tensor/tensor_view.h"
#include "model_fused_kernels.h"

namespace inferx::kernels {
namespace {

class CudaMallocDomain final : public AllocationDomain {
 public:
  absl::Status Release(void* address, ByteCount, ByteCount, AllocationId) override {
    cudaFree(address);
    return absl::OkStatus();
  }
  void Abandon(void* address, ByteCount, ByteCount, AllocationId) noexcept override {
    cudaFree(address);
  }
};

template <typename T>
struct Tensor {
  Buffer buffer;
  MutableTensorView mutable_view;
  TensorView view;
};

template <typename T>
absl::StatusOr<Tensor<T>> MakeTensor(std::span<const uint64_t> dimensions, Dtype dtype,
                                     std::span<const T> values, bool device) {
  auto shape = Shape::Create(dimensions);
  if (!shape.ok()) return shape.status();
  auto strides = Strides::Contiguous(*shape);
  if (!strides.ok()) return strides.status();
  auto bytes = shape->Bytes(dtype);
  if (!bytes.ok()) return bytes.status();
  if (device) {
    void* memory = nullptr;
    if (cudaMalloc(&memory, bytes->value()) != cudaSuccess) {
      return absl::ResourceExhaustedError("cudaMalloc failed");
    }
    if (!values.empty()) {
      cudaMemcpy(memory, values.data(), bytes->value(), cudaMemcpyHostToDevice);
    }
    auto buffer = Buffer::Adopt(
        memory,
        AllocationRequest{Device::Cuda(DeviceId(0)), MemoryKind::kDevice, *bytes, ByteCount(256),
                          MemoryCategory::kTest},
        ByteCount(256),
        [] {
          auto id = NextAllocationId();
          return id.ok() ? *id : AllocationId(0);
        }(),
        std::make_shared<CudaMallocDomain>());
    if (!buffer.ok()) return buffer.status();
    auto view = buffer->MutableView(ByteRange{ByteCount(0), *bytes});
    if (!view.ok()) return view.status();
    auto tensor = MutableTensorView::Create(*view, dtype, *shape, *strides);
    if (!tensor.ok()) return tensor.status();
    return Tensor<T>{std::move(*buffer), *tensor, tensor->AsConst()};
  }
  CpuAllocator allocator;
  auto buffer = allocator.Allocate(AllocationRequest{Device::Host(), MemoryKind::kHost, *bytes,
                                                     ByteCount(alignof(T)), MemoryCategory::kTest});
  if (!buffer.ok()) return buffer.status();
  auto view = buffer->MutableView(ByteRange{ByteCount(0), *bytes});
  if (!view.ok()) return view.status();
  auto host = view->HostBytes();
  if (!host.ok()) return host.status();
  if (!values.empty()) std::memcpy(host->data(), values.data(), host->size());
  auto tensor = MutableTensorView::Create(*view, dtype, *shape, *strides);
  if (!tensor.ok()) return tensor.status();
  return Tensor<T>{std::move(*buffer), *tensor, tensor->AsConst()};
}

template <typename T>
Tensor<T> Required(std::initializer_list<uint64_t> dimensions, Dtype dtype,
                   std::span<const T> values, bool device) {
  const std::vector<uint64_t> dims(dimensions);
  auto tensor = MakeTensor<T>(std::span<const uint64_t>(dims), dtype, values, device);
  if (!tensor.ok()) {
    ADD_FAILURE() << tensor.status();
    std::abort();
  }
  return std::move(*tensor);
}

template <typename T>
std::vector<T> ReadBack(const TensorView& view) {
  const void* address = ::inferx::kernels::BufferAccess::Address(view.buffer());
  const size_t count = view.buffer().size().value() / sizeof(T);
  std::vector<T> result(count);
  if (address != nullptr && count != 0) {
    cudaMemcpy(result.data(), address, count * sizeof(T), cudaMemcpyDeviceToHost);
  }
  return result;
}

template <typename T>
std::span<const T> HostValues(const Tensor<T>& tensor) {
  auto bytes = tensor.view.buffer().HostBytes();
  return std::span<const T>(reinterpret_cast<const T*>(bytes->data()), bytes->size() / sizeof(T));
}

std::vector<float> RandomFloats(size_t count, uint32_t seed, float low = -1.5F, float high = 1.5F) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(low, high);
  std::vector<float> values(count);
  for (float& value : values) value = dist(rng);
  return values;
}

KernelExecutionContext Ctx(void* workspace = nullptr, uint64_t bytes = 0) {
  KernelExecutionContext context;
  context.device = Device::Cuda(DeviceId(0));
  context.stream_handle = 0;
  context.compute_capability = 89;
  context.workspace = workspace;
  context.workspace_bytes = bytes;
  return context;
}

// Host FP8 e4m3 decoder for test verification.
float Fp8BitsToFloat(uint8_t bits) {
  const float sign = (bits & 0x80U) != 0 ? -1.0F : 1.0F;
  const uint32_t exponent = (bits >> 3) & 0x0FU;
  const uint32_t mantissa = bits & 0x07U;
  if (exponent == 0) {
    return sign * static_cast<float>(mantissa) * (1.0F / 8.0F) * (1.0F / 64.0F);
  }
  if (exponent == 0x0FU && mantissa == 0x07U) return std::numeric_limits<float>::quiet_NaN();
  return sign * (1.0F + static_cast<float>(mantissa) / 8.0F) *
         std::pow(2.0F, static_cast<float>(exponent) - 7.0F);
}

struct NewOpsFixture : public ::testing::Test {
  void SetUp() override {
    ResetKernelBackendsForTest();
    ASSERT_TRUE(cuda::RegisterCudaKernelBackend().ok());
  }
  void TearDown() override {
    ResetKernelBackendsForTest();
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
};

TEST_F(NewOpsFixture, FlashInferActMulSiluMatchesReference) {
  constexpr uint64_t kTokens = 33;
  constexpr uint64_t kHalf = 96;
  const std::vector<float> fused_input = RandomFloats(kTokens * 2 * kHalf, 5);
  const std::vector<float> zeros(kTokens * kHalf, 0.0F);
  auto device_input = Required<float>({kTokens, 2 * kHalf}, Dtype::kFloat32, fused_input, true);
  auto device_output = Required<float>({kTokens, kHalf}, Dtype::kFloat32, zeros, true);
  ASSERT_TRUE(
      LaunchActMul({device_input.view, device_output.mutable_view, ActMulKind::kSilu}, Ctx()).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto host_input = Required<float>({kTokens, 2 * kHalf}, Dtype::kFloat32, fused_input, false);
  auto host_output = Required<float>({kTokens, kHalf}, Dtype::kFloat32, zeros, false);
  ASSERT_TRUE(ReferenceActMul({host_input.view, host_output.mutable_view, ActMulKind::kSilu}).ok());
  const std::vector<float> actual = ReadBack<float>(device_output.view);
  const std::span<const float> expected = HostValues(host_output);
  for (size_t index = 0; index < actual.size(); ++index) {
    ASSERT_NEAR(actual[index], expected[index], 2.0e-4F) << index;
  }
}

TEST_F(NewOpsFixture, FlashInferFusedAddRmsNormMatchesReference) {
  constexpr uint64_t kTokens = 9;
  constexpr uint64_t kHidden = 128;
  const std::vector<float> input = RandomFloats(kTokens * kHidden, 6);
  const std::vector<float> residual = RandomFloats(kTokens * kHidden, 7);
  const std::vector<float> weight = RandomFloats(kHidden, 8);
  auto device_input = Required<float>({kTokens, kHidden}, Dtype::kFloat32, input, true);
  auto device_residual = Required<float>({kTokens, kHidden}, Dtype::kFloat32, residual, true);
  auto device_weight = Required<float>({kHidden}, Dtype::kFloat32, weight, true);
  FusedAddRmsNormRequest request{device_input.mutable_view, device_residual.mutable_view,
                                 device_weight.view, 1.0e-5F};
  ASSERT_TRUE(LaunchFusedAddRmsNorm(request, Ctx()).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto host_input = Required<float>({kTokens, kHidden}, Dtype::kFloat32, input, false);
  auto host_residual = Required<float>({kTokens, kHidden}, Dtype::kFloat32, residual, false);
  auto host_weight = Required<float>({kHidden}, Dtype::kFloat32, weight, false);
  FusedAddRmsNormRequest host_request{host_input.mutable_view, host_residual.mutable_view,
                                      host_weight.view, 1.0e-5F};
  ASSERT_TRUE(ReferenceFusedAddRmsNorm(host_request).ok());
  const std::vector<float> actual = ReadBack<float>(device_input.view);
  const std::span<const float> expected = HostValues(host_input);
  for (size_t index = 0; index < actual.size(); ++index) {
    ASSERT_NEAR(actual[index], expected[index], 2.0e-4F) << index;
  }
  const std::vector<float> actual_residual = ReadBack<float>(device_residual.view);
  const std::span<const float> expected_residual = HostValues(host_residual);
  for (size_t index = 0; index < actual_residual.size(); ++index) {
    ASSERT_NEAR(actual_residual[index], expected_residual[index], 1.0e-5F) << index;
  }
}

TEST_F(NewOpsFixture, FlashInferQkRmsNormMatchesReference) {
  constexpr uint64_t kTokens = 4;
  constexpr uint64_t kQueryHeads = 8;
  constexpr uint64_t kKvHeads = 2;
  constexpr uint64_t kHeadDim = 64;
  const std::vector<float> query = RandomFloats(kTokens * kQueryHeads * kHeadDim, 9);
  const std::vector<float> key = RandomFloats(kTokens * kKvHeads * kHeadDim, 10);
  const std::vector<float> q_weight = RandomFloats(kHeadDim, 11, 0.5F, 1.5F);
  const std::vector<float> k_weight = RandomFloats(kHeadDim, 12, 0.5F, 1.5F);
  auto device_q = Required<float>({kTokens, kQueryHeads, kHeadDim}, Dtype::kFloat32, query, true);
  auto device_k = Required<float>({kTokens, kKvHeads, kHeadDim}, Dtype::kFloat32, key, true);
  auto device_qw = Required<float>({kHeadDim}, Dtype::kFloat32, q_weight, true);
  auto device_kw = Required<float>({kHeadDim}, Dtype::kFloat32, k_weight, true);
  QkRmsNormRequest request{device_q.mutable_view, device_k.mutable_view, device_qw.view,
                           device_kw.view, 1.0e-5F};
  ASSERT_TRUE(LaunchQkRmsNorm(request, Ctx()).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto host_q = Required<float>({kTokens, kQueryHeads, kHeadDim}, Dtype::kFloat32, query, false);
  auto host_k = Required<float>({kTokens, kKvHeads, kHeadDim}, Dtype::kFloat32, key, false);
  auto host_qw = Required<float>({kHeadDim}, Dtype::kFloat32, q_weight, false);
  auto host_kw = Required<float>({kHeadDim}, Dtype::kFloat32, k_weight, false);
  QkRmsNormRequest host_request{host_q.mutable_view, host_k.mutable_view, host_qw.view,
                                host_kw.view, 1.0e-5F};
  ASSERT_TRUE(ReferenceQkRmsNorm(host_request).ok());
  for (const auto& [actual, expected, tensor] :
       {std::tuple{ReadBack<float>(device_q.view), HostValues(host_q), "q"},
        std::tuple{ReadBack<float>(device_k.view), HostValues(host_k), "k"}}) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < actual.size(); ++index) {
      ASSERT_NEAR(actual[index], expected[index], 2.0e-4F) << tensor << " " << index;
    }
  }
}

TEST_F(NewOpsFixture, ArgmaxAndTopKRenormMatchReference) {
  constexpr uint64_t kRows = 6;
  constexpr uint64_t kVocab = 1024;
  std::vector<float> logits = RandomFloats(kRows * kVocab, 13, -8.0F, 8.0F);
  logits[0] = std::numeric_limits<float>::quiet_NaN();  // NaN row -> -1
  logits[1] = std::numeric_limits<float>::quiet_NaN();
  auto device_logits = Required<float>({kRows, kVocab}, Dtype::kFloat32, logits, true);
  auto device_ids = Required<int32_t>({kRows}, Dtype::kInt32, std::span<const int32_t>{}, true);
  ASSERT_TRUE(LaunchArgmax({device_logits.view, device_ids.mutable_view}, Ctx()).ok());
  auto host_logits = Required<float>({kRows, kVocab}, Dtype::kFloat32, logits, false);
  auto host_ids = Required<int32_t>({kRows}, Dtype::kInt32, std::vector<int32_t>(kRows), false);
  ASSERT_TRUE(ReferenceArgmax({host_logits.view, host_ids.mutable_view}).ok());
  const std::vector<int32_t> actual = ReadBack<int32_t>(device_ids.view);
  const std::span<const int32_t> expected = HostValues(host_ids);
  for (size_t row = 0; row < kRows; ++row) {
    EXPECT_EQ(actual[row], expected[row]) << "row " << row;
  }

  // top-k renorm on a fresh probability tensor
  std::vector<float> probs_raw = RandomFloats(kRows * kVocab, 14, 0.0F, 0.01F);
  auto device_probs = Required<float>({kRows, kVocab}, Dtype::kFloat32, probs_raw, true);
  ASSERT_TRUE(LaunchTopKRenorm({device_probs.mutable_view, 8}, Ctx()).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto host_probs = Required<float>({kRows, kVocab}, Dtype::kFloat32, probs_raw, false);
  ASSERT_TRUE(ReferenceTopKRenorm({host_probs.mutable_view, 8}).ok());
  const std::vector<float> actual_p = ReadBack<float>(device_probs.view);
  const std::span<const float> expected_p = HostValues(host_probs);
  for (size_t index = 0; index < actual_p.size(); ++index) {
    ASSERT_NEAR(actual_p[index], expected_p[index], 1.0e-6F) << index;
  }
}

TEST_F(NewOpsFixture, Fp8QuantQuantizesWithinHalfUlp) {
  constexpr uint64_t kTokens = 5;
  constexpr uint64_t kDim = 256;
  const std::vector<float> input = RandomFloats(kTokens * kDim, 15, -3.0F, 3.0F);
  auto device_input = Required<float>({kTokens, kDim}, Dtype::kFloat32, input, true);
  auto device_output =
      Required<uint8_t>({kTokens, kDim}, Dtype::kUInt8, std::span<const uint8_t>{}, true);
  auto device_scales = Required<float>({kTokens}, Dtype::kFloat32, std::span<const float>{}, true);
  Fp8QuantRequest request{device_input.view, device_output.mutable_view, device_scales.mutable_view,
                          QuantGranularity::kToken, 128};
  ASSERT_TRUE(LaunchFp8Quant(request, Ctx()).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const std::vector<float> scales = ReadBack<float>(device_scales.view);
  const std::vector<uint8_t> packed = ReadBack<uint8_t>(device_output.view);
  ASSERT_EQ(scales.size(), kTokens);
  ASSERT_EQ(packed.size(), kTokens * kDim);
  for (uint64_t token = 0; token < kTokens; ++token) {
    const float scale = scales[token];
    ASSERT_GT(scale, 0.0F);
    for (uint64_t index = 0; index < kDim; ++index) {
      const float value = input[token * kDim + index];
      const float quantized = Fp8BitsToFloat(packed[token * kDim + index]) * scale;
      // e4m3 mantissa (3 bits): a correctly rounded value is within half a
      // quantization step of the scaled input.
      // e4m3 step at magnitude |q| in [2^k, 2^(k+1)) is 2^k/8, so a
      // correctly rounded value is within 2^k/16 <= |q|/16 of the input.
      const float tolerance = std::fabs(quantized) / 15.0F + scale * 0.51F + 1.0e-6F;
      if (std::fabs(quantized - value) > tolerance) {
        ADD_FAILURE() << "token " << token << " index " << index << " bits "
                      << static_cast<unsigned>(packed[token * kDim + index]) << " decoded "
                      << quantized << " value " << value << " scale " << scale;
        return;
      }
    }
  }
}

TEST_F(NewOpsFixture, SoftmaxTopKMatchesReference) {
  constexpr uint64_t kTokens = 7;
  constexpr uint64_t kExperts = 64;
  constexpr uint32_t kTopK = 8;
  const std::vector<float> logits = RandomFloats(kTokens * kExperts, 16, -6.0F, 6.0F);
  auto device_logits = Required<float>({kTokens, kExperts}, Dtype::kFloat32, logits, true);
  auto device_weights =
      Required<float>({kTokens, kTopK}, Dtype::kFloat32, std::span<const float>{}, true);
  auto device_ids =
      Required<int32_t>({kTokens, kTopK}, Dtype::kInt32, std::span<const int32_t>{}, true);
  SoftmaxTopKRequest request{device_logits.view, device_weights.mutable_view,
                             device_ids.mutable_view, true};
  ASSERT_TRUE(LaunchSoftmaxTopK(request, Ctx()).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto host_logits = Required<float>({kTokens, kExperts}, Dtype::kFloat32, logits, false);
  auto host_weights = Required<float>({kTokens, kTopK}, Dtype::kFloat32,
                                      std::vector<float>(kTokens * kTopK), false);
  auto host_ids = Required<int32_t>({kTokens, kTopK}, Dtype::kInt32,
                                    std::vector<int32_t>(kTokens * kTopK), false);
  ASSERT_TRUE(ReferenceSoftmaxTopK(
                  {host_logits.view, host_weights.mutable_view, host_ids.mutable_view, true})
                  .ok());
  const std::vector<int32_t> actual_ids = ReadBack<int32_t>(device_ids.view);
  const std::span<const int32_t> expected_ids = HostValues(host_ids);
  const std::vector<float> actual_w = ReadBack<float>(device_weights.view);
  const std::span<const float> expected_w = HostValues(host_weights);
  for (size_t index = 0; index < actual_ids.size(); ++index) {
    ASSERT_EQ(actual_ids[index], expected_ids[index]) << index;
    ASSERT_NEAR(actual_w[index], expected_w[index], 1.0e-5F) << index;
  }
}

TEST_F(NewOpsFixture, HadamardTransformMatchesReference) {
  constexpr uint64_t kRows = 24;
  const std::vector<float> input = RandomFloats(kRows * 128, 17);
  const std::vector<float> zeros(kRows * 128, 0.0F);
  auto device_input = Required<float>({kRows, 128}, Dtype::kFloat32, input, true);
  auto device_output = Required<float>({kRows, 128}, Dtype::kFloat32, zeros, true);
  const absl::Status hadamard_status =
      LaunchHadamardTransform({device_input.view, device_output.mutable_view, 0.05F}, Ctx());
  ASSERT_TRUE(hadamard_status.ok()) << hadamard_status;
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto host_input = Required<float>({kRows, 128}, Dtype::kFloat32, input, false);
  auto host_output = Required<float>({kRows, 128}, Dtype::kFloat32, zeros, false);
  ASSERT_TRUE(ReferenceHadamardTransform({host_input.view, host_output.mutable_view, 0.05F}).ok());
  const std::vector<float> actual = ReadBack<float>(device_output.view);
  const std::span<const float> expected = HostValues(host_output);
  for (size_t index = 0; index < actual.size(); ++index) {
    ASSERT_NEAR(actual[index], expected[index], 5.0e-3F) << index;
  }
}

TEST_F(NewOpsFixture, AttnResMatchesReference) {
  constexpr uint64_t kTokens = 5;
  constexpr uint64_t kHidden = 128;
  constexpr uint64_t kBlocks = 3;
  const std::vector<float> layer = RandomFloats(kTokens * kHidden, 18);
  const std::vector<float> blocks = RandomFloats(kBlocks * kTokens * kHidden, 19);
  const std::vector<float> res_w = RandomFloats(kHidden, 20, 0.5F, 1.5F);
  const std::vector<float> rms_w = RandomFloats(kHidden, 21, 0.5F, 1.5F);
  auto device_layer = Required<float>({kTokens, kHidden}, Dtype::kFloat32, layer, true);
  auto device_blocks = Required<float>({kBlocks, kTokens, kHidden}, Dtype::kFloat32, blocks, true);
  auto device_res_w = Required<float>({kHidden}, Dtype::kFloat32, res_w, true);
  auto device_rms_w = Required<float>({kHidden}, Dtype::kFloat32, rms_w, true);
  AttnResRequest request{device_layer.mutable_view,
                         device_blocks.view,
                         device_res_w.view,
                         device_rms_w.view,
                         std::nullopt,
                         1.0e-5F,
                         1.0e-5F};
  const absl::Status attn_res_status = LaunchAttnRes(request, Ctx());
  ASSERT_TRUE(attn_res_status.ok()) << attn_res_status;
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto host_layer = Required<float>({kTokens, kHidden}, Dtype::kFloat32, layer, false);
  auto host_blocks = Required<float>({kBlocks, kTokens, kHidden}, Dtype::kFloat32, blocks, false);
  auto host_res_w = Required<float>({kHidden}, Dtype::kFloat32, res_w, false);
  auto host_rms_w = Required<float>({kHidden}, Dtype::kFloat32, rms_w, false);
  AttnResRequest host_request{host_layer.mutable_view,
                              host_blocks.view,
                              host_res_w.view,
                              host_rms_w.view,
                              std::nullopt,
                              1.0e-5F,
                              1.0e-5F};
  ASSERT_TRUE(ReferenceAttnRes(host_request).ok());
  const std::vector<float> actual = ReadBack<float>(device_layer.view);
  const std::span<const float> expected = HostValues(host_layer);
  for (size_t index = 0; index < actual.size(); ++index) {
    ASSERT_NEAR(actual[index], expected[index], 2.0e-3F) << index;
  }
}

TEST_F(NewOpsFixture, HcMixAndCombineMatchReference) {
  constexpr uint64_t kTokens = 3;
  constexpr uint32_t kHc = 2;
  constexpr uint32_t kHidden = 64;
  constexpr uint32_t kRank = 32;
  const uint64_t kWide = kHc * kHidden;
  const std::vector<float> normalized = RandomFloats(kTokens * kWide, 22);
  const std::vector<float> proj_w = RandomFloats((kRank + kHc) * kWide, 23, -0.05F, 0.05F);
  const std::vector<float> up_w = RandomFloats(kWide * kRank, 24, -0.05F, 0.05F);
  void* workspace = nullptr;
  cudaMalloc(&workspace, cuda::hc::HcMixWorkspaceBytes(kTokens, kRank, kHc, kHidden));
  auto device_normalized = Required<float>({kTokens, kWide}, Dtype::kFloat32, normalized, true);
  auto device_proj_w = Required<float>({kRank + kHc, kWide}, Dtype::kFloat32, proj_w, true);
  auto device_up_w = Required<float>({kWide, kRank}, Dtype::kFloat32, up_w, true);
  auto device_mixed = Required<float>({kTokens, kHidden}, Dtype::kFloat32,
                                      std::vector<float>(kTokens * kHidden), true);
  auto device_inject =
      Required<float>({kTokens, kHc}, Dtype::kFloat32, std::vector<float>(kTokens * kHc), true);
  HcMixRequest mix_request{device_normalized.view,
                           device_proj_w.view,
                           device_up_w.view,
                           device_mixed.mutable_view,
                           device_inject.mutable_view,
                           kHc,
                           kHidden,
                           kRank,
                           1.0F};
  const absl::Status hc_status = LaunchHcMix(
      mix_request, Ctx(workspace, cuda::hc::HcMixWorkspaceBytes(kTokens, kRank, kHc, kHidden)));
  ASSERT_TRUE(hc_status.ok()) << hc_status;
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto host_normalized = Required<float>({kTokens, kWide}, Dtype::kFloat32, normalized, false);
  auto host_proj_w = Required<float>({kRank + kHc, kWide}, Dtype::kFloat32, proj_w, false);
  auto host_up_w = Required<float>({kWide, kRank}, Dtype::kFloat32, up_w, false);
  auto host_mixed = Required<float>({kTokens, kHidden}, Dtype::kFloat32,
                                    std::vector<float>(kTokens * kHidden), false);
  auto host_inject =
      Required<float>({kTokens, kHc}, Dtype::kFloat32, std::vector<float>(kTokens * kHc), false);
  HcMixRequest host_mix{host_normalized.view,
                        host_proj_w.view,
                        host_up_w.view,
                        host_mixed.mutable_view,
                        host_inject.mutable_view,
                        kHc,
                        kHidden,
                        kRank,
                        1.0F};
  ASSERT_TRUE(ReferenceHcMix(host_mix).ok());
  const std::vector<float> actual_mixed = ReadBack<float>(device_mixed.view);
  const std::span<const float> expected_mixed = HostValues(host_mixed);
  for (size_t index = 0; index < actual_mixed.size(); ++index) {
    ASSERT_NEAR(actual_mixed[index], expected_mixed[index], 5.0e-3F) << index;
  }
  const std::vector<float> actual_inject = ReadBack<float>(device_inject.view);
  const std::span<const float> expected_inject = HostValues(host_inject);
  for (size_t index = 0; index < actual_inject.size(); ++index) {
    ASSERT_NEAR(actual_inject[index], expected_inject[index], 2.0e-3F) << index;
  }
  cudaFree(workspace);
}

TEST_F(NewOpsFixture, MhcPreAndPostMatchReference) {
  constexpr uint64_t kTokens = 2;
  constexpr uint32_t kStreams = 2;
  constexpr uint32_t kHidden = 48;
  const uint64_t kWide = kStreams * kHidden;
  const uint64_t kMixRows = 2 * kStreams + kStreams * kStreams;
  const std::vector<float> residual = RandomFloats(kTokens * kWide, 25);
  const std::vector<float> fn = RandomFloats(kMixRows * kWide, 26, -0.02F, 0.02F);
  const std::vector<float> hc_scale = {1.0F, 1.0F, 1.0F};
  const std::vector<float> hc_base = RandomFloats(kMixRows, 27, -0.1F, 0.1F);
  void* workspace = nullptr;
  const uint64_t workspace_bytes = cuda::mhc::MhcPreWorkspaceBytes(kTokens, kStreams);
  cudaMalloc(&workspace, workspace_bytes);
  auto device_residual =
      Required<float>({kTokens, kStreams, kHidden}, Dtype::kFloat32, residual, true);
  auto device_fn = Required<float>({kMixRows, kWide}, Dtype::kFloat32, fn, true);
  auto device_scale = Required<float>({3}, Dtype::kFloat32, hc_scale, true);
  auto device_base = Required<float>({kMixRows}, Dtype::kFloat32, hc_base, true);
  auto device_layer = Required<float>({kTokens, kHidden}, Dtype::kFloat32,
                                      std::vector<float>(kTokens * kHidden), true);
  auto device_post = Required<float>({kTokens, kStreams}, Dtype::kFloat32,
                                     std::vector<float>(kTokens * kStreams), true);
  auto device_comb = Required<float>({kTokens, kStreams, kStreams}, Dtype::kFloat32,
                                     std::vector<float>(kTokens * kStreams * kStreams), true);
  MhcPreRequest pre{device_residual.view,
                    device_fn.view,
                    device_scale.view,
                    device_base.view,
                    device_layer.mutable_view,
                    device_post.mutable_view,
                    device_comb.mutable_view,
                    1.0e-5F,
                    1.0e-5F,
                    4};
  ASSERT_TRUE(LaunchMhcPre(pre, Ctx(workspace, workspace_bytes)).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto host_residual =
      Required<float>({kTokens, kStreams, kHidden}, Dtype::kFloat32, residual, false);
  auto host_fn = Required<float>({kMixRows, kWide}, Dtype::kFloat32, fn, false);
  auto host_scale = Required<float>({3}, Dtype::kFloat32, hc_scale, false);
  auto host_base = Required<float>({kMixRows}, Dtype::kFloat32, hc_base, false);
  auto host_layer = Required<float>({kTokens, kHidden}, Dtype::kFloat32,
                                    std::vector<float>(kTokens * kHidden), false);
  auto host_post = Required<float>({kTokens, kStreams}, Dtype::kFloat32,
                                   std::vector<float>(kTokens * kStreams), false);
  auto host_comb = Required<float>({kTokens, kStreams, kStreams}, Dtype::kFloat32,
                                   std::vector<float>(kTokens * kStreams * kStreams), false);
  MhcPreRequest host_pre{host_residual.view,
                         host_fn.view,
                         host_scale.view,
                         host_base.view,
                         host_layer.mutable_view,
                         host_post.mutable_view,
                         host_comb.mutable_view,
                         1.0e-5F,
                         1.0e-5F,
                         4};
  ASSERT_TRUE(ReferenceMhcPre(host_pre).ok());
  const std::vector<float> actual_layer = ReadBack<float>(device_layer.view);
  const std::span<const float> expected_layer = HostValues(host_layer);
  for (size_t index = 0; index < actual_layer.size(); ++index) {
    ASSERT_NEAR(actual_layer[index], expected_layer[index], 5.0e-3F) << index;
  }
  const std::vector<float> actual_post = ReadBack<float>(device_post.view);
  const std::span<const float> expected_post = HostValues(host_post);
  for (size_t index = 0; index < actual_post.size(); ++index) {
    ASSERT_NEAR(actual_post[index], expected_post[index], 2.0e-3F) << index;
  }
  cudaFree(workspace);
}

}  // namespace
}  // namespace inferx::kernels
