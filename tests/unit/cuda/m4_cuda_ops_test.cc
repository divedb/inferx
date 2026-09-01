#include <cuda_runtime_api.h>

#include <cstdint>

#include "gtest/gtest.h"
#include "inferx/ops/kernel_registry.h"
#include "inferx/platform/cuda/ops/cublaslt_gemm.h"
#include "inferx/platform/cuda/ops/cuda_kernel_registry.h"
#include "inferx/platform/cuda/ops/flashinfer_attention.h"

namespace inferx::cuda::ops {
namespace {

TEST(M4CudaOpsTest, RegistersExactSmCapabilitiesAndCreatesCublasLtContext) {
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  ASSERT_GT(device_count, 0) << "the required M4 GPU lane may not pass by skipping";
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
  const uint16_t compute_capability =
      static_cast<uint16_t>(properties.major * 10 + properties.minor);

  inferx::ops::KernelRegistry registry(16);
  ASSERT_TRUE(RegisterCudaCapabilities(registry, compute_capability).ok());
  ASSERT_TRUE(registry.Freeze().ok());
  EXPECT_EQ(registry.size(), 8);

  auto context = CublasLtContext::Create(DeviceId(0));
  ASSERT_TRUE(context.ok()) << context.status();
  EXPECT_TRUE(context->Close().ok());
  EXPECT_EQ(FlashInferAvailability().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace inferx::cuda::ops
