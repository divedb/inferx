#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "test_kernels.h"

namespace inferx::cuda::test {
namespace {

__global__ void StridedCopyKernel(const std::byte* source, std::byte* destination,
                                  StridedCopyParams params) {
  const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= params.element_count) return;
  uint64_t remaining = linear;
  uint64_t source_element = 0;
  uint64_t destination_element = 0;
  for (uint8_t reverse_axis = params.rank; reverse_axis > 0; --reverse_axis) {
    const uint8_t axis = static_cast<uint8_t>(reverse_axis - 1);
    const uint64_t extent = params.dimensions[axis];
    const uint64_t index = extent == 0 ? 0 : remaining % extent;
    if (extent != 0) remaining /= extent;
    source_element += index * params.source_strides[axis];
    destination_element += index * params.destination_strides[axis];
  }
  const uint64_t source_offset = params.source_byte_offset + source_element * params.element_size;
  const uint64_t destination_offset =
      params.destination_byte_offset + destination_element * params.element_size;
  for (uint32_t byte = 0; byte < params.element_size; ++byte) {
    destination[destination_offset + byte] = source[source_offset + byte];
  }
}

}  // namespace

void LaunchStridedCopy(const void* source, void* destination, StridedCopyParams params,
                       cudaStream_t stream) {
  if (params.element_count == 0) return;
  constexpr uint32_t kThreads = 256;
  const uint64_t blocks = (params.element_count + kThreads - 1) / kThreads;
  StridedCopyKernel<<<static_cast<uint32_t>(blocks), kThreads, 0, stream>>>(
      static_cast<const std::byte*>(source), static_cast<std::byte*>(destination), params);
}

}  // namespace inferx::cuda::test
