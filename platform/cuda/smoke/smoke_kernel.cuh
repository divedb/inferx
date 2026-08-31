// Kernel launch entry shared between the .cu TU and the host smoke driver.
#ifndef INFERX_PLATFORM_CUDA_SMOKE_SMOKE_KERNEL_CUH_
#define INFERX_PLATFORM_CUDA_SMOKE_SMOKE_KERNEL_CUH_

void launch_inferx_smoke_write_pattern(int* buffer, int count);

#endif  // INFERX_PLATFORM_CUDA_SMOKE_SMOKE_KERNEL_CUH_
