// Sampling category launch shims: FlashInfer top-p renormalization plus
// owned argmax and top-k renormalization (no external provider at the pin).
#ifndef INFERX_KERNELS_CUDA_SAMPLING_KERNELS_H_
#define INFERX_KERNELS_CUDA_SAMPLING_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/storage_type.h"

namespace inferx::kernels::cuda::sampling {

// Top-p renormalization in place over probs[rows, vocab] (FlashInfer kernel).
cudaError_t LaunchFlashInferTopPRenorm(void* probs, uint64_t rows, uint64_t vocab, float top_p,
                                       StorageType dtype, cudaStream_t stream);
// Top-k renormalization in place (owned kernel; threshold binary search).
cudaError_t LaunchTopKRenorm(void* probs, uint64_t rows, uint64_t vocab, uint32_t top_k,
                             StorageType dtype, cudaStream_t stream);
// NaN-safe argmax with ties to the lowest index; all-NaN rows yield -1
// (owned kernel).
cudaError_t LaunchArgmax(const void* logits, uint64_t rows, uint64_t vocab, int32_t* output,
                         StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::sampling

#endif  // INFERX_KERNELS_CUDA_SAMPLING_KERNELS_H_
