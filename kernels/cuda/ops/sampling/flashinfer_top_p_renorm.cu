// Top-p renormalization via FlashInfer's device kernel (ADR 0032 reuse
// tier 2). In-place over probs[rows, vocab].
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "flashinfer/sampling.cuh"
#include "inferx/kernels/cuda/sampling_kernels.h"

namespace inferx::kernels::cuda::sampling {
namespace {

constexpr uint32_t kBlockThreads = 256;
constexpr cub::BlockReduceAlgorithm kReduceAlgo = cub::BLOCK_REDUCE_WARP_REDUCTIONS;
constexpr uint32_t kVecSize = 4;

cudaError_t LaunchTyped(void* probs, uint64_t rows, uint64_t vocab, float top_p,
                        cudaStream_t stream) {
  flashinfer::sampling::TopPRenormProbKernel<kBlockThreads, kReduceAlgo, kVecSize, float>
      <<<static_cast<uint32_t>(rows), kBlockThreads,
         sizeof(flashinfer::sampling::RenormTempStorage<kBlockThreads, kReduceAlgo>), stream>>>(
          static_cast<float*>(probs), static_cast<float*>(probs),
          /*top_p_arr=*/nullptr, top_p, static_cast<uint32_t>(vocab));
  return cudaGetLastError();
}

}  // namespace

cudaError_t LaunchFlashInferTopPRenorm(void* probs, uint64_t rows, uint64_t vocab, float top_p,
                                       StorageType dtype, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  // The pinned kernel's reduction helpers instantiate for FP32 only.
  if (dtype != StorageType::kFloat32) return cudaErrorInvalidValue;
  return LaunchTyped(probs, rows, vocab, top_p, stream);
}

}  // namespace inferx::kernels::cuda::sampling
