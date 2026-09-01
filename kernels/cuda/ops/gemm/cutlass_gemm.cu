// Dense GEMM/logits via CUTLASS (ADR 0032 last-resort tier).
//
// Justification recorded in ADR 0032: hpc-ops GEMMs are FP8-only and SM90+;
// flashinfer's CUTLASS GEMM runner is BF16-only and requires its vendored
// CUTLASS closure. Neither serves the FP32/FP16/BF16 row-major contract, so
// this is the maintained custom implementation. FP32 uses SIMT (full FP32
// math); FP16/BF16 use Ampere tensor-op MMA with FP32 accumulation. CUTLASS
// 4.x requires cutlass::half_t/bfloat16_t element types for arch::Mma
// specializations (bit-compatible with __half/__nv_bfloat16).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "inferx/kernels/cuda/gemm_kernels.h"

namespace inferx::kernels::cuda::gemm {
namespace {

using cutlass::layout::ColumnMajor;
using cutlass::layout::RowMajor;

// output[m, n] = alpha * sum_k input[m, k] * weight[n, k] + beta * addend.
// A = input row-major [M, K]; B = weight row-major [N, K] consumed as the
// column-major [K, N] operand; C/D row-major [M, N].
template <typename T, typename OpClass, typename ArchTag, typename Accum>
struct GemmConfig {
  using Gemm = cutlass::gemm::device::Gemm<T, RowMajor, T, ColumnMajor, T, RowMajor, Accum,
                                           OpClass, ArchTag>;
};

template <typename Gemm, typename Element>
cudaError_t RunGemm(const void* input, const void* weight, const void* addend, void* output,
                    uint64_t tokens, uint64_t in_dim, uint64_t out_dim, float alpha, float beta,
                    cudaStream_t stream) {
  using Arguments = typename Gemm::Arguments;
  const void* source = addend != nullptr ? addend : output;
  Element* c_ptr = static_cast<Element*>(const_cast<void*>(source));
  const int m = static_cast<int>(tokens);
  const int n = static_cast<int>(out_dim);
  const int k = static_cast<int>(in_dim);
  const float used_beta = addend != nullptr ? beta : 0.0F;
  const Arguments args(
      {m, n, k},
      {static_cast<const Element*>(input), k},
      {static_cast<const Element*>(weight), k},
      {c_ptr, n},
      {static_cast<Element*>(output), n},
      {alpha, used_beta});
  Gemm op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
  if (op.initialize(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
  if (op(stream) != cutlass::Status::kSuccess) return cudaErrorUnknown;
  return cudaPeekAtLastError();
}

}  // namespace

cudaError_t LaunchCutlassGemm(const void* input, const void* weight, const void* addend,
                              void* output, uint64_t tokens, uint64_t in_dim, uint64_t out_dim,
                              float alpha, float beta, StorageType dtype, cudaStream_t stream) {
  if (tokens == 0 || in_dim == 0 || out_dim == 0) return cudaSuccess;
  if (tokens > INT32_MAX || in_dim > INT32_MAX || out_dim > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  switch (dtype) {
    case StorageType::kFloat32: {
      using Fp32Gemm = typename GemmConfig<float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
                                           float>::Gemm;
      return RunGemm<Fp32Gemm, float>(input, weight, addend, output, tokens, in_dim, out_dim,
                                      alpha, beta, stream);
    }
    case StorageType::kFloat16: {
      using Fp16Gemm = typename GemmConfig<cutlass::half_t, cutlass::arch::OpClassTensorOp,
                                           cutlass::arch::Sm80, float>::Gemm;
      return RunGemm<Fp16Gemm, cutlass::half_t>(input, weight, addend, output, tokens, in_dim,
                                                out_dim, alpha, beta, stream);
    }
    case StorageType::kBFloat16: {
      using Bf16Gemm = typename GemmConfig<cutlass::bfloat16_t, cutlass::arch::OpClassTensorOp,
                                           cutlass::arch::Sm80, float>::Gemm;
      return RunGemm<Bf16Gemm, cutlass::bfloat16_t>(input, weight, addend, output, tokens, in_dim,
                                                    out_dim, alpha, beta, stream);
    }
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace inferx::kernels::cuda::gemm
