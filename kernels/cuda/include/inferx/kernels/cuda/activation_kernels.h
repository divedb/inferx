// Activation category launch shims (owned kernels).
#ifndef INFERX_KERNELS_CUDA_ACTIVATION_KERNELS_H_
#define INFERX_KERNELS_CUDA_ACTIVATION_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/storage_type.h"

namespace inferx::kernels::cuda::activation {

cudaError_t LaunchSwiGluKernel(const void* gate, const void* up, void* output,
                               uint64_t elements, StorageType dtype, cudaStream_t stream);
cudaError_t LaunchResidualKernel(const void* left, const void* right, void* output,
                                 uint64_t elements, StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::activation

#endif  // INFERX_KERNELS_CUDA_ACTIVATION_KERNELS_H_
