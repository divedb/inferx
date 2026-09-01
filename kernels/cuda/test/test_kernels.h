#ifndef INFERX_KERNELS_CUDA_TEST_TEST_KERNELS_H_
#define INFERX_KERNELS_CUDA_TEST_TEST_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

namespace inferx::cuda::test {

struct StridedCopyParams {
  uint64_t dimensions[8]{};
  uint64_t source_strides[8]{};
  uint64_t destination_strides[8]{};
  uint64_t source_byte_offset = 0;
  uint64_t destination_byte_offset = 0;
  uint64_t element_count = 0;
  uint32_t element_size = 0;
  uint8_t rank = 0;
};

void LaunchStridedCopy(const void* source, void* destination, StridedCopyParams params,
                       cudaStream_t stream);
void LaunchSynchronizationProbe(uint32_t* destination, cudaStream_t stream);
void LaunchBoundedDelay(uint64_t clock_cycles, cudaStream_t stream);
void LaunchIllegalAddress(cudaStream_t stream);
void LaunchDeviceAssert(cudaStream_t stream);

}  // namespace inferx::cuda::test

#endif  // INFERX_KERNELS_CUDA_TEST_TEST_KERNELS_H_
