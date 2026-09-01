#include <cuda_runtime.h>

#include <cstdint>

#include "test_kernels.h"

namespace inferx::cuda::test {
namespace {

__global__ void BoundedDelayKernel(uint64_t clock_cycles) {
  const uint64_t begin = clock64();
  while (clock64() - begin < clock_cycles) {
  }
}

}  // namespace

void LaunchBoundedDelay(uint64_t clock_cycles, cudaStream_t stream) {
  BoundedDelayKernel<<<1, 1, 0, stream>>>(clock_cycles);
}

}  // namespace inferx::cuda::test
