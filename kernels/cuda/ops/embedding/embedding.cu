// Embedding gather: last-resort custom kernel (ADR 0032 provider audit).
//
// Neither hpc-ops nor flashinfer ships an embedding lookup at the pinned
// revisions (both treat it as a host/library concern). The host mirror is the
// validation oracle; the kernel re-checks IDs so a mutated device vector can
// only produce a bounded read, never an out-of-bounds write.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/embedding_kernels.h"
#include "inferx/kernels/cuda/kernel_utils.cuh"

namespace inferx::kernels::cuda::embedding {
namespace {

template <typename T>
__global__ void EmbeddingKernel(const int32_t* __restrict__ ids, const T* __restrict__ weight,
                                T* __restrict__ output, uint64_t tokens, uint64_t vocabulary,
                                uint64_t hidden) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t elements = tokens * hidden;
  if (index >= elements) return;
  const int32_t id = ids[index / hidden];
  if (id >= 0 && static_cast<uint64_t>(id) < vocabulary) {
    output[index] = weight[static_cast<uint64_t>(id) * hidden + index % hidden];
  }
}

}  // namespace

cudaError_t LaunchEmbeddingKernel(const int32_t* ids, const void* weight, void* output,
                                  uint64_t tokens, uint64_t vocabulary, uint64_t hidden,
                                  StorageType dtype, cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  uint64_t work = 0;
  cudaError_t status = CheckedWork(tokens, hidden, &work);
  uint32_t blocks = 0;
  if (status == cudaSuccess) status = CheckedBlocks(work, &blocks);
  if (status != cudaSuccess) return status;
  switch (dtype) {
    case StorageType::kFloat32:
      EmbeddingKernel<float><<<blocks, kKernelThreads, 0, stream>>>(
          ids, static_cast<const float*>(weight), static_cast<float*>(output), tokens, vocabulary,
          hidden);
      break;
    case StorageType::kFloat16:
      EmbeddingKernel<__half><<<blocks, kKernelThreads, 0, stream>>>(
          ids, static_cast<const __half*>(weight), static_cast<__half*>(output), tokens, vocabulary,
          hidden);
      break;
    case StorageType::kBFloat16:
      EmbeddingKernel<__nv_bfloat16><<<blocks, kKernelThreads, 0, stream>>>(
          ids, static_cast<const __nv_bfloat16*>(weight), static_cast<__nv_bfloat16*>(output),
          tokens, vocabulary, hidden);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

}  // namespace inferx::kernels::cuda::embedding
