#include <cuda_runtime.h>

#include <cstdint>

#include "test_kernels.h"

namespace inferx::cuda::test {
namespace {

__global__ void SynchronizationProbeKernel(uint32_t* destination) {
  __shared__ uint32_t values[32];
  const uint32_t lane = threadIdx.x;
  values[lane] = lane + 1;
  __syncthreads();
  if (lane == 0) {
    uint32_t sum = 0;
    for (uint32_t index = 0; index < 32; ++index) sum += values[index];
    *destination = sum;
  }
}

}  // namespace

void LaunchSynchronizationProbe(uint32_t* destination, cudaStream_t stream) {
  SynchronizationProbeKernel<<<1, 32, 0, stream>>>(destination);
}

}  // namespace inferx::cuda::test
