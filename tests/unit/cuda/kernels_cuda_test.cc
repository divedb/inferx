// GPU tests for the kernels-architecture CUDA backend (ADR 0031/0032):
// provider-chain selection on SM89, forced-provider failures, and oracle
// execution of the flashinfer RMSNorm/RoPE wrappers, the CUTLASS GEMM, and
// the last-resort custom kernels against the CPU reference executors.
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "cuda_kernel_backend.h"  // kernels/cuda/dispatch (public via target)
#include "gtest/gtest.h"
#include "inferx/kernels/cuda/buffer_access.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/ops/activation.h"
#include "inferx/kernels/ops/gemm.h"
#include "inferx/kernels/ops/layernorm.h"
#include "inferx/kernels/ops/transform.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/ops/rope.h"
#include "inferx/ops/swiglu.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::kernels {
namespace {

// Device memory adopted into a Buffer without going through the platform
// layer: cudaMalloc plus a domain whose Release/Abandon return it.
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
struct OwnedTensor {
  Buffer buffer;
  MutableTensorView mutable_view;
  TensorView view;
};

template <typename T>
absl::StatusOr<OwnedTensor<T>> MakeTensor(std::initializer_list<uint64_t> dimensions, Dtype dtype,
                                          std::span<const T> values, Device device,
                                          MemoryKind memory_kind) {
  const std::vector<uint64_t> dims(dimensions);
  return MakeTensor<T>(std::span<const uint64_t>(dims), dtype, values, device, memory_kind);
}

template <typename T>
absl::StatusOr<OwnedTensor<T>> MakeTensor(std::span<const uint64_t> dimensions, Dtype dtype,
                                          std::span<const T> values, Device device,
                                          MemoryKind memory_kind) {
  auto shape = Shape::Create(dimensions);
  if (!shape.ok()) return shape.status();
  auto strides = Strides::Contiguous(*shape);
  if (!strides.ok()) return strides.status();
  auto bytes = shape->Bytes(dtype);
  if (!bytes.ok()) return bytes.status();

  void* allocation = nullptr;
  if (device.kind == DeviceKind::kCuda) {
    if (cudaMalloc(&allocation, bytes->value()) != cudaSuccess) {
      return absl::ResourceExhaustedError("cudaMalloc failed");
    }
    if (!values.empty()) {
      cudaMemcpy(allocation, values.data(), bytes->value(), cudaMemcpyHostToDevice);
    }
    auto buffer = Buffer::Adopt(
        allocation,
        AllocationRequest{device, memory_kind, *bytes, ByteCount(256), MemoryCategory::kTest},
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
    return OwnedTensor<T>{std::move(*buffer), *tensor, tensor->AsConst()};
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
  return OwnedTensor<T>{std::move(*buffer), *tensor, tensor->AsConst()};
}

template <typename T>
OwnedTensor<T> Required(std::span<const uint64_t> dimensions, Dtype dtype,
                        std::span<const T> values, Device device, MemoryKind memory) {
  auto tensor = MakeTensor<T>(dimensions, dtype, values, device, memory);
  if (!tensor.ok()) ADD_FAILURE() << tensor.status();
  OwnedTensor<T> empty{};
  return tensor.ok() ? std::move(*tensor) : std::move(empty);
}

// Tensor construction failures (allocation exhaustion, invalid shapes) are
// not recoverable inside a GPU oracle test: report and abort this process.
template <typename T>
OwnedTensor<T> Required(std::initializer_list<uint64_t> dimensions, Dtype dtype,
                        std::span<const T> values, Device device, MemoryKind memory) {
  const std::vector<uint64_t> dims(dimensions);
  auto tensor = MakeTensor<T>(std::span<const uint64_t>(dims), dtype, values, device, memory);
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
std::span<const T> HostValues(const OwnedTensor<T>& tensor) {
  auto bytes = tensor.view.buffer().HostBytes();
  return std::span<const T>(reinterpret_cast<const T*>(bytes->data()), bytes->size() / sizeof(T));
}

KernelExecutionContext Context89() {
  KernelExecutionContext context;
  context.device = Device::Cuda(DeviceId(0));
  context.stream_handle = 0;  // default stream
  context.compute_capability = 89;
  return context;
}

struct KernelsCudaFixture : public ::testing::Test {
  void SetUp() override {
    ResetKernelBackendsForTest();
    ASSERT_TRUE(cuda::RegisterCudaKernelBackend().ok());
  }
  void TearDown() override {
    ResetKernelBackendsForTest();
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
};

TEST_F(KernelsCudaFixture, ChainSelectsReuseProvidersFirstOnSm89) {
  KernelBackend* backend = FindKernelBackend(DeviceKind::kCuda);
  ASSERT_NE(backend, nullptr);
  auto rms = backend->SelectProvider(ops::OpKind::kRmsNorm, 89, std::nullopt);
  ASSERT_TRUE(rms.ok()) << rms.status();
  EXPECT_EQ(*rms, ProviderId::kFlashInfer);
  auto rope = backend->SelectProvider(ops::OpKind::kRope, 89, std::nullopt);
  ASSERT_TRUE(rope.ok()) << rope.status();
  EXPECT_EQ(*rope, ProviderId::kFlashInfer);
  auto gemm = backend->SelectProvider(ops::OpKind::kGemm, 89, std::nullopt);
  ASSERT_TRUE(gemm.ok()) << gemm.status();
  EXPECT_EQ(*gemm, ProviderId::kCutlass);
  auto embedding = backend->SelectProvider(ops::OpKind::kEmbedding, 89, std::nullopt);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  EXPECT_EQ(*embedding, ProviderId::kInferxOwned);
  auto swiglu = backend->SelectProvider(ops::OpKind::kSiluMultiply, 89, std::nullopt);
  ASSERT_TRUE(swiglu.ok()) << swiglu.status();
  EXPECT_EQ(*swiglu, ProviderId::kInferxOwned);
}

TEST_F(KernelsCudaFixture, AttentionSelectsFlashInferOnSm89) {
  KernelBackend* backend = FindKernelBackend(DeviceKind::kCuda);
  const auto attention = backend->SelectProvider(ops::OpKind::kAttention, 89, std::nullopt);
  ASSERT_TRUE(attention.ok()) << attention.status();
  EXPECT_EQ(*attention, ProviderId::kFlashInfer);
}

TEST_F(KernelsCudaFixture, ForcedHpcOpsGemmIsUnimplemented) {
  KernelBackend* backend = FindKernelBackend(DeviceKind::kCuda);
  const auto forced = backend->SelectProvider(ops::OpKind::kGemm, 90, ProviderId::kHpcOps);
  EXPECT_EQ(forced.status().code(), absl::StatusCode::kUnimplemented);
}

std::vector<float> RandomFloats(size_t count, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.5F, 1.5F);
  std::vector<float> values(count);
  for (float& value : values) value = dist(rng);
  return values;
}

TEST_F(KernelsCudaFixture, FlashInferRmsNormMatchesReference) {
  constexpr uint64_t kTokens = 17;
  constexpr uint64_t kHidden = 64;
  const std::vector<float> input = RandomFloats(kTokens * kHidden, 42);
  const std::vector<float> weight = RandomFloats(kHidden, 43);
  const std::vector<float> zeros(kTokens * kHidden, 0.0F);

  auto device_input = Required<float>({kTokens, kHidden}, Dtype::kFloat32, input,
                                      Device::Cuda(DeviceId(0)), MemoryKind::kDevice);
  auto device_weight = Required<float>({kHidden}, Dtype::kFloat32, weight,
                                       Device::Cuda(DeviceId(0)), MemoryKind::kDevice);
  auto device_output = Required<float>({kTokens, kHidden}, Dtype::kFloat32, zeros,
                                       Device::Cuda(DeviceId(0)), MemoryKind::kDevice);
  ops::RmsNormRequest request{device_input.view, device_weight.view, device_output.mutable_view,
                              1.0e-5F};
  ASSERT_TRUE(LaunchRmsNorm(request, Context89()).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto host_input = Required<float>({kTokens, kHidden}, Dtype::kFloat32, input, Device::Host(),
                                    MemoryKind::kHost);
  auto host_weight =
      Required<float>({kHidden}, Dtype::kFloat32, weight, Device::Host(), MemoryKind::kHost);
  auto host_output = Required<float>({kTokens, kHidden}, Dtype::kFloat32, zeros, Device::Host(),
                                     MemoryKind::kHost);
  ASSERT_TRUE(
      ops::ReferenceRmsNorm({host_input.view, host_weight.view, host_output.mutable_view, 1.0e-5F})
          .ok());

  const std::vector<float> actual = ReadBack<float>(device_output.view);
  const std::span<const float> expected = HostValues(host_output);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    ASSERT_NEAR(actual[index], expected[index], 2.0e-4F) << "element " << index;
  }
}

TEST_F(KernelsCudaFixture, CustomSwiGluMatchesReference) {
  constexpr uint64_t kTokens = 9;
  constexpr uint64_t kHalf = 32;
  const std::vector<float> gate = RandomFloats(kTokens * kHalf, 7);
  const std::vector<float> up = RandomFloats(kTokens * kHalf, 8);
  const std::vector<float> zeros(kTokens * kHalf, 0.0F);

  auto device_gate = Required<float>({kTokens, kHalf}, Dtype::kFloat32, gate,
                                     Device::Cuda(DeviceId(0)), MemoryKind::kDevice);
  auto device_up = Required<float>({kTokens, kHalf}, Dtype::kFloat32, up, Device::Cuda(DeviceId(0)),
                                   MemoryKind::kDevice);
  auto device_output = Required<float>({kTokens, kHalf}, Dtype::kFloat32, zeros,
                                       Device::Cuda(DeviceId(0)), MemoryKind::kDevice);
  ASSERT_TRUE(
      LaunchSwiGlu({device_gate.view, device_up.view, device_output.mutable_view}, Context89())
          .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto host_gate =
      Required<float>({kTokens, kHalf}, Dtype::kFloat32, gate, Device::Host(), MemoryKind::kHost);
  auto host_up =
      Required<float>({kTokens, kHalf}, Dtype::kFloat32, up, Device::Host(), MemoryKind::kHost);
  auto host_output =
      Required<float>({kTokens, kHalf}, Dtype::kFloat32, zeros, Device::Host(), MemoryKind::kHost);
  ASSERT_TRUE(ops::ReferenceSwiGlu({host_gate.view, host_up.view, host_output.mutable_view}).ok());

  const std::vector<float> actual = ReadBack<float>(device_output.view);
  const std::span<const float> expected = HostValues(host_output);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    ASSERT_NEAR(actual[index], expected[index], 1.0e-5F) << "element " << index;
  }
}

TEST_F(KernelsCudaFixture, CutlassGemmMatchesReference) {
  constexpr uint64_t kTokens = 12;
  constexpr uint64_t kIn = 48;
  constexpr uint64_t kOut = 32;
  const std::vector<float> input = RandomFloats(kTokens * kIn, 11);
  const std::vector<float> weight = RandomFloats(kOut * kIn, 12);
  const std::vector<float> zeros(kTokens * kOut, 0.0F);

  auto device_input = Required<float>({kTokens, kIn}, Dtype::kFloat32, input,
                                      Device::Cuda(DeviceId(0)), MemoryKind::kDevice);
  auto device_weight = Required<float>({kOut, kIn}, Dtype::kFloat32, weight,
                                       Device::Cuda(DeviceId(0)), MemoryKind::kDevice);
  auto device_output = Required<float>({kTokens, kOut}, Dtype::kFloat32, zeros,
                                       Device::Cuda(DeviceId(0)), MemoryKind::kDevice);
  ops::GemmRequest request{device_input.view, device_weight.view, std::nullopt,
                           device_output.mutable_view};
  ASSERT_TRUE(LaunchGemm(request, Context89()).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto host_input =
      Required<float>({kTokens, kIn}, Dtype::kFloat32, input, Device::Host(), MemoryKind::kHost);
  auto host_weight =
      Required<float>({kOut, kIn}, Dtype::kFloat32, weight, Device::Host(), MemoryKind::kHost);
  auto host_output =
      Required<float>({kTokens, kOut}, Dtype::kFloat32, zeros, Device::Host(), MemoryKind::kHost);
  ASSERT_TRUE(ops::ReferenceGemm(
                  {host_input.view, host_weight.view, std::nullopt, host_output.mutable_view})
                  .ok());

  const std::vector<float> actual = ReadBack<float>(device_output.view);
  const std::span<const float> expected = HostValues(host_output);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    ASSERT_NEAR(actual[index], expected[index], 2.0e-3F) << "element " << index;
  }
}

TEST_F(KernelsCudaFixture, FlashInferRopeMatchesReference) {
  constexpr uint64_t kTokens = 5;
  constexpr uint64_t kQueryHeads = 8;
  constexpr uint64_t kKvHeads = 4;
  constexpr uint64_t kHeadDim = 64;
  constexpr uint64_t kMaxPosition = 512;
  const std::vector<float> query = RandomFloats(kTokens * kQueryHeads * kHeadDim, 21);
  const std::vector<float> key = RandomFloats(kTokens * kKvHeads * kHeadDim, 22);
  const std::vector<float> zeros_q(kTokens * kQueryHeads * kHeadDim, 0.0F);
  const std::vector<float> zeros_k(kTokens * kKvHeads * kHeadDim, 0.0F);
  std::vector<int32_t> positions(kTokens);
  for (uint64_t token = 0; token < kTokens; ++token)
    positions[token] = static_cast<int32_t>(7 + token);

  const Device cuda = Device::Cuda(DeviceId(0));
  auto device_query = Required<float>({kTokens, kQueryHeads, kHeadDim}, Dtype::kFloat32, query,
                                      cuda, MemoryKind::kDevice);
  auto device_key = Required<float>({kTokens, kKvHeads, kHeadDim}, Dtype::kFloat32, key, cuda,
                                    MemoryKind::kDevice);
  auto device_positions =
      Required<int32_t>({kTokens}, Dtype::kInt32, positions, cuda, MemoryKind::kDevice);
  auto device_query_out = Required<float>({kTokens, kQueryHeads, kHeadDim}, Dtype::kFloat32,
                                          zeros_q, cuda, MemoryKind::kDevice);
  auto device_key_out = Required<float>({kTokens, kKvHeads, kHeadDim}, Dtype::kFloat32, zeros_k,
                                        cuda, MemoryKind::kDevice);
  ops::RopeRequest request{device_query.view,
                           device_key.view,
                           device_positions.view,
                           device_query_out.mutable_view,
                           device_key_out.mutable_view,
                           10'000.0F,
                           kMaxPosition,
                           std::span(positions)};
  ASSERT_TRUE(LaunchRope(request, Context89()).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto host_query = Required<float>({kTokens, kQueryHeads, kHeadDim}, Dtype::kFloat32, query,
                                    Device::Host(), MemoryKind::kHost);
  auto host_key = Required<float>({kTokens, kKvHeads, kHeadDim}, Dtype::kFloat32, key,
                                  Device::Host(), MemoryKind::kHost);
  auto host_positions =
      Required<int32_t>({kTokens}, Dtype::kInt32, positions, Device::Host(), MemoryKind::kHost);
  auto host_query_out = Required<float>({kTokens, kQueryHeads, kHeadDim}, Dtype::kFloat32, zeros_q,
                                        Device::Host(), MemoryKind::kHost);
  auto host_key_out = Required<float>({kTokens, kKvHeads, kHeadDim}, Dtype::kFloat32, zeros_k,
                                      Device::Host(), MemoryKind::kHost);
  ASSERT_TRUE(ops::ReferenceRope({host_query.view, host_key.view, host_positions.view,
                                  host_query_out.mutable_view, host_key_out.mutable_view, 10'000.0F,
                                  kMaxPosition, std::span(positions)})
                  .ok());

  const std::vector<float> actual_q = ReadBack<float>(device_query_out.view);
  const std::span<const float> expected_q = HostValues(host_query_out);
  ASSERT_EQ(actual_q.size(), expected_q.size());
  for (size_t index = 0; index < actual_q.size(); ++index) {
    ASSERT_NEAR(actual_q[index], expected_q[index], 2.0e-3F) << "q element " << index;
  }
  const std::vector<float> actual_k = ReadBack<float>(device_key_out.view);
  const std::span<const float> expected_k = HostValues(host_key_out);
  for (size_t index = 0; index < actual_k.size(); ++index) {
    ASSERT_NEAR(actual_k[index], expected_k[index], 2.0e-3F) << "k element " << index;
  }
}

}  // namespace
}  // namespace inferx::kernels
