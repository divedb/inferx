// Activation category launch shims (owned kernels).
#ifndef INFERX_KERNELS_CUDA_ACTIVATION_KERNELS_H_
#define INFERX_KERNELS_CUDA_ACTIVATION_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/storage_type.h"

namespace inferx::kernels::cuda::activation {

// FlashInfer act_and_mul_kernel instantiation: input [tokens, 2*d] fused
// gate|up layout, output [tokens, d]; kind selects silu/gelu/gelu_tanh.
cudaError_t LaunchActMulKernel(const void* input, void* output, uint64_t tokens, uint64_t half_dim,
                               uint32_t kind, StorageType dtype, cudaStream_t stream);
// 3-way residual add.
cudaError_t LaunchAdd3Kernel(const void* a, const void* b, const void* c, void* output,
                             uint64_t elements, StorageType dtype, cudaStream_t stream);
cudaError_t LaunchSwiGluKernel(const void* gate, const void* up, void* output, uint64_t elements,
                               StorageType dtype, cudaStream_t stream);
cudaError_t LaunchResidualKernel(const void* left, const void* right, void* output,
                                 uint64_t elements, StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::activation

#endif  // INFERX_KERNELS_CUDA_ACTIVATION_KERNELS_H_
