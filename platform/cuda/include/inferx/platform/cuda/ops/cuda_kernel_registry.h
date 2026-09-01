#ifndef INFERX_PLATFORM_CUDA_OPS_CUDA_KERNEL_REGISTRY_H_
#define INFERX_PLATFORM_CUDA_OPS_CUDA_KERNEL_REGISTRY_H_

#include "absl/status/status.h"
#include "inferx/ops/embedding.h"
#include "inferx/ops/kernel_registry.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/ops/rope.h"
#include "inferx/ops/swiglu.h"
#include "inferx/platform/cuda/ops/cuda_op_context.h"

namespace inferx::cuda::ops {

[[nodiscard]] absl::Status RegisterCudaCapabilities(inferx::ops::KernelRegistry& registry,
                                                    uint16_t compute_capability);
[[nodiscard]] absl::Status LaunchEmbedding(const inferx::ops::EmbeddingRequest& request,
                                           const CudaOpContext& context);
[[nodiscard]] absl::Status LaunchRmsNorm(const inferx::ops::RmsNormRequest& request,
                                         const CudaOpContext& context);
[[nodiscard]] absl::Status LaunchRope(const inferx::ops::RopeRequest& request,
                                      const CudaOpContext& context);
[[nodiscard]] absl::Status LaunchSwiGlu(const inferx::ops::SwiGluRequest& request,
                                        const CudaOpContext& context);
[[nodiscard]] absl::Status LaunchResidual(const inferx::ops::ResidualRequest& request,
                                          const CudaOpContext& context);

}  // namespace inferx::cuda::ops

#endif  // INFERX_PLATFORM_CUDA_OPS_CUDA_KERNEL_REGISTRY_H_
