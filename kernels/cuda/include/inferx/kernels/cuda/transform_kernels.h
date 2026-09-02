// Transform category launch shim (FlashInfer RoPE instantiation).
#ifndef INFERX_KERNELS_CUDA_TRANSFORM_KERNELS_H_
#define INFERX_KERNELS_CUDA_TRANSFORM_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/storage_type.h"

namespace inferx::kernels::cuda::transform {

// FlashInfer's kernel takes head_dim as a compile-time parameter: dims
// {32, 64, 128, 256} are instantiated; others return cudaErrorInvalidValue
// here and surface as Unimplemented from the provider layer.
cudaError_t LaunchFlashInferRope(const void* query, const void* key, const int32_t* positions,
                                 void* query_output, void* key_output, uint64_t tokens,
                                 uint64_t query_heads, uint64_t kv_heads, uint64_t head_dimension,
                                 float theta, StorageType dtype, cudaStream_t stream);

// Length-128 Hadamard transform (owned kernel).
cudaError_t LaunchHadamard128(const void* input, void* output, uint64_t rows, float scale,
                              StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::transform

#endif  // INFERX_KERNELS_CUDA_TRANSFORM_KERNELS_H_
