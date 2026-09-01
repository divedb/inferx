#include <cuda_runtime.h>

#include "test_kernels.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>

namespace inferx::cuda::test {
namespace {

__global__ void IllegalAddressKernel() {
  auto* invalid = reinterpret_cast<volatile int*>(static_cast<uintptr_t>(1));
  *invalid = 7;
}

__global__ void DeviceAssertKernel() { assert(false); }

}  // namespace

void LaunchIllegalAddress(cudaStream_t stream) { IllegalAddressKernel<<<1, 1, 0, stream>>>(); }

void LaunchDeviceAssert(cudaStream_t stream) { DeviceAssertKernel<<<1, 1, 0, stream>>>(); }

}  // namespace inferx::cuda::test
