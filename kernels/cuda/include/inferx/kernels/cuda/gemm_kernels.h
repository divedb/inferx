// GEMM category launch shim (custom CUTLASS implementation; ADR 0032).
//
// Computes output[tokens, out] = alpha * input[tokens, in] x weight^T + beta *
// addend (optional), with FP32 accumulation for every storage type.
#ifndef INFERX_KERNELS_CUDA_GEMM_KERNELS_H_
#define INFERX_KERNELS_CUDA_GEMM_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/storage_type.h"

namespace inferx::kernels::cuda::gemm {

cudaError_t LaunchCutlassGemm(const void* input, const void* weight, const void* addend,
                              void* output, uint64_t tokens, uint64_t in_dim, uint64_t out_dim,
                              float alpha, float beta, StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::gemm

#endif  // INFERX_KERNELS_CUDA_GEMM_KERNELS_H_
