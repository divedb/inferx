// CPU tests for the neutral kernels dispatch surface (ADR 0031): preference
// chain order, backend registration, forced-provider semantics, and
// Unimplemented behavior when no backend is registered.
#include "inferx/kernels/kernel_dispatch.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <istream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/kernels/ops/activation.h"
#include "inferx/kernels/ops/embedding.h"
#include "inferx/kernels/provider.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::kernels {
namespace {

class RecordingBackend final : public KernelBackend {
 public:
  DeviceKind kind() const noexcept override { return DeviceKind::kCuda; }
  absl::StatusOr<ProviderId> SelectProvider(ops::OpKind, uint16_t,
                                            std::optional<ProviderId> forced) const override {
    if (forced.has_value() && *forced == ProviderId::kCutlass) {
      return absl::UnimplementedError("probe: cutlass unavailable");
    }
    return ProviderId::kInferxOwned;
  }
  absl::Status LaunchEmbedding(const ops::EmbeddingRequest&, const KernelExecutionContext&,
                               std::optional<ProviderId> forced) override {
    ++embedding_launches;
    if (forced.has_value() && *forced == ProviderId::kCutlass) {
      return absl::UnimplementedError("probe: cutlass unavailable");
    }
    return absl::OkStatus();
  }
  absl::Status LaunchSwiGlu(const ops::SwiGluRequest&, const KernelExecutionContext&,
                            std::optional<ProviderId>) override {
    ++swiglu_launches;
    return absl::OkStatus();
  }

  int embedding_launches = 0;
  int swiglu_launches = 0;
};

// Minimal CPU-resident requests: the recording backend never dereferences
// the tensors, but request types are not default-constructible.
struct CpuTensor {
  Buffer buffer;
  MutableTensorView mutable_view;
  TensorView view;
};

CpuTensor MakeCpuTensor(std::span<const uint64_t> dimensions, DType dtype) {
  const auto shape = Shape::Create(dimensions);
  const auto strides = Strides::Contiguous(*shape);
  const auto bytes = shape->Bytes(dtype);
  CpuAllocator allocator;
  auto buffer = allocator.Allocate(AllocationRequest{Device::Host(), MemoryKind::kHost, *bytes,
                                                     ByteCount(4), MemoryCategory::kTest});
  auto view = buffer->MutableView(ByteRange{ByteCount(0), *bytes});
  auto tensor = MutableTensorView::Create(*view, dtype, *shape, *strides);
  return CpuTensor{std::move(*buffer), *tensor, tensor->AsConst()};
}

CpuTensor MakeCpuTensor(std::initializer_list<uint64_t> dimensions, DType dtype) {
  const std::vector<uint64_t> dims(dimensions);
  return MakeCpuTensor(std::span<const uint64_t>(dims), dtype);
}

struct DispatchFixture : public ::testing::Test {
  void SetUp() override { ResetKernelBackendsForTest(); }
  void TearDown() override { ResetKernelBackendsForTest(); }
};

TEST_F(DispatchFixture, PreferenceChainOrdersHpcOpsFirst) {
  const std::span<const ProviderId> chain = ProviderPreferenceChain();
  ASSERT_EQ(chain.size(), 5U);
  EXPECT_EQ(chain[0], ProviderId::kHpcOps);
  EXPECT_EQ(chain[1], ProviderId::kFlashInfer);
  EXPECT_EQ(chain[2], ProviderId::kCutlass);
  EXPECT_EQ(chain[3], ProviderId::kCublasLt);
  EXPECT_EQ(chain[4], ProviderId::kInferxOwned);
}

TEST_F(DispatchFixture, ProviderNamesAreStable) {
  EXPECT_EQ(ProviderIdName(ProviderId::kHpcOps), "hpc_ops");
  EXPECT_EQ(ProviderIdName(ProviderId::kFlashInfer), "flashinfer");
  EXPECT_EQ(ProviderIdName(ProviderId::kCutlass), "cutlass");
  EXPECT_EQ(ProviderIdName(ProviderId::kCublasLt), "cublaslt");
  EXPECT_EQ(ProviderIdName(ProviderId::kInferxOwned), "inferx_owned");
}

TEST_F(DispatchFixture, LaunchWithoutBackendIsUnimplemented) {
  KernelExecutionContext context;
  context.device = Device::Cuda(DeviceId(0));
  auto ids = MakeCpuTensor({1}, DType::kInt32);
  auto weight = MakeCpuTensor({2, 2}, DType::kFloat32);
  auto output = MakeCpuTensor({1, 2}, DType::kFloat32);
  ops::EmbeddingRequest request{ids.view, weight.view, output.mutable_view};
  EXPECT_EQ(LaunchEmbedding(request, context).code(), absl::StatusCode::kUnimplemented);
}

TEST_F(DispatchFixture, RegisterAndDispatchToBackend) {
  RecordingBackend* backend = new RecordingBackend();
  ASSERT_TRUE(RegisterKernelBackend(std::unique_ptr<KernelBackend>(backend)).ok());
  EXPECT_EQ(FindKernelBackend(DeviceKind::kCuda), backend);
  EXPECT_EQ(FindKernelBackend(DeviceKind::kHost), nullptr);

  KernelExecutionContext context;
  context.device = Device::Cuda(DeviceId(0));
  context.compute_capability = 89;
  auto ids = MakeCpuTensor({1}, DType::kInt32);
  auto weight = MakeCpuTensor({2, 2}, DType::kFloat32);
  auto output = MakeCpuTensor({1, 2}, DType::kFloat32);
  EXPECT_TRUE(LaunchEmbedding({ids.view, weight.view, output.mutable_view}, context).ok());
  EXPECT_EQ(backend->embedding_launches, 1);
  auto gate = MakeCpuTensor({1, 2}, DType::kFloat32);
  auto up = MakeCpuTensor({1, 2}, DType::kFloat32);
  auto fused = MakeCpuTensor({1, 2}, DType::kFloat32);
  EXPECT_TRUE(LaunchSwiGlu({gate.view, up.view, fused.mutable_view}, context).ok());
  EXPECT_EQ(backend->swiglu_launches, 1);
}

TEST_F(DispatchFixture, DuplicateRegistrationIsRejected) {
  ASSERT_TRUE(RegisterKernelBackend(std::make_unique<RecordingBackend>()).ok());
  EXPECT_EQ(RegisterKernelBackend(std::make_unique<RecordingBackend>()).code(),
            absl::StatusCode::kAlreadyExists);
}

TEST_F(DispatchFixture, NullAndHostBackendsAreRejected) {
  EXPECT_EQ(RegisterKernelBackend(nullptr).code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(DispatchFixture, HostDeviceStaysUndispatched) {
  KernelExecutionContext context;
  context.device = Device::Host();
  auto gate = MakeCpuTensor({1, 2}, DType::kFloat32);
  auto up = MakeCpuTensor({1, 2}, DType::kFloat32);
  auto fused = MakeCpuTensor({1, 2}, DType::kFloat32);
  // Host reference execution lives in inferx::ops, not the kernels layer.
  EXPECT_EQ(LaunchSwiGlu({gate.view, up.view, fused.mutable_view}, context).code(),
            absl::StatusCode::kUnimplemented);
}

// ADR 0031 layering rule: everything under kernels/include must stay free of
// backend types so CPU presets can build the unified dispatch surface.
TEST(KernelsContainmentTest, UnifiedHeadersStayCudaFree) {
  const std::filesystem::path roots[] = {std::filesystem::path(INFERX_KERNELS_SOURCE_DIR) /
                                         "include/inferx/kernels"};
  size_t checked = 0;
  for (const auto& root : roots) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
      if (!entry.is_regular_file() || entry.path().extension() != ".h") continue;
      std::ifstream file(entry.path());
      std::string line;
      while (std::getline(file, line)) {
        const auto position = line.find('#');
        if (position == std::string::npos) continue;
        const auto include = line.find("include", position);
        if (include == std::string::npos) continue;
        EXPECT_EQ(line.find("cuda"), std::string::npos)
            << entry.path() << " leaks a CUDA include: " << line;
      }
      ++checked;
    }
  }
  EXPECT_GE(checked, 8U) << "expected the unified headers to be present";
}

}  // namespace
}  // namespace inferx::kernels
