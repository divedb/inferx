// Embedding category launch shim (owned kernel).
#ifndef INFERX_KERNELS_CUDA_EMBEDDING_KERNELS_H_
#define INFERX_KERNELS_CUDA_EMBEDDING_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/storage_type.h"

namespace inferx::kernels::cuda::embedding {

cudaError_t LaunchEmbeddingKernel(const int32_t* ids, const void* weight, void* output,
                                  uint64_t tokens, uint64_t vocabulary, uint64_t hidden,
                                  StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::embedding

#endif  // INFERX_KERNELS_CUDA_EMBEDDING_KERNELS_H_
