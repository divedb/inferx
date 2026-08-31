// Minimal device kernel for the M0 CUDA smoke: writes a known pattern that
// the host validates element-by-element (m0.md section 10.2).
#include "smoke_kernel.cuh"

__global__ void inferx_smoke_write_pattern(int* buffer, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    buffer[index] = index * 7 + 13;
  }
}

void launch_inferx_smoke_write_pattern(int* buffer, int count) {
  // Tiny buffer: one block is enough and keeps the launch deterministic.
  inferx_smoke_write_pattern<<<1, count>>>(buffer, count);
}
