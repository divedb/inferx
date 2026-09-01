// Per-operator request validation shared by every CUDA provider (ADR 0031).
// Providers layer their own availability envelopes (Available) on top; these
// checks define what is structurally launchable on a CUDA device at all.
#ifndef INFERX_KERNELS_CUDA_DISPATCH_OP_VALIDATION_H_
#define INFERX_KERNELS_CUDA_DISPATCH_OP_VALIDATION_H_

#include "absl/status/status.h"
#include "cuda_launch_context.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/embedding.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/logits.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/ops/rope.h"
#include "inferx/ops/swiglu.h"

namespace inferx::kernels::cuda {

absl::Status ValidateEmbeddingForCuda(const ops::EmbeddingRequest& request,
                                      const CudaLaunchContext& context);
absl::Status ValidateRmsNormForCuda(const ops::RmsNormRequest& request,
                                    const CudaLaunchContext& context);
absl::Status ValidateRopeForCuda(const ops::RopeRequest& request,
                                 const CudaLaunchContext& context);
absl::Status ValidateSwiGluForCuda(const ops::SwiGluRequest& request,
                                   const CudaLaunchContext& context);
absl::Status ValidateResidualForCuda(const ops::ResidualRequest& request,
                                     const CudaLaunchContext& context);
absl::Status ValidateGemmForCuda(const ops::GemmRequest& request,
                                 const CudaLaunchContext& context);
absl::Status ValidateLogitsForCuda(const ops::LogitsRequest& request,
                                   const CudaLaunchContext& context);
absl::Status ValidateAttentionForCuda(const ops::AttentionRequest& request,
                                      const DeviceAttentionMetadata& metadata,
                                      const CudaLaunchContext& context);

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_DISPATCH_OP_VALIDATION_H_
