// Per-op launch functions for the kernels-layer contracts (ADR 0031
// taxonomy): validate, marshal, call the category shim, and wrap the result.
// The provider layer (custom/flashinfer) binds these via SimpleOpProvider.
#ifndef INFERX_KERNELS_CUDA_DISPATCH_OP_LAUNCHERS_H_
#define INFERX_KERNELS_CUDA_DISPATCH_OP_LAUNCHERS_H_

#include <cstdint>

#include "absl/status/status.h"
#include "cuda_launch_context.h"
#include "inferx/kernels/ops/activation.h"
#include "inferx/kernels/ops/layernorm.h"
#include "inferx/kernels/ops/model_fused.h"
#include "inferx/kernels/ops/sampling.h"
#include "inferx/kernels/ops/transform.h"

namespace inferx::kernels::cuda::launchers {

// Each launcher returns absl::Status; availability probes return whether a
// request (null = chain probe) is servable at a compute capability.
absl::Status ActMul(const ActMulRequest& request, const CudaLaunchContext& context);
bool ActMulAvailable(const ActMulRequest* probe, uint16_t compute_capability);
absl::Status Add3(const Add3Request& request, const CudaLaunchContext& context);
bool Add3Available(const Add3Request* probe, uint16_t compute_capability);
absl::Status FusedAddRmsNorm(const FusedAddRmsNormRequest& request,
                             const CudaLaunchContext& context);
bool FusedAddRmsNormAvailable(const FusedAddRmsNormRequest* probe, uint16_t compute_capability);
absl::Status GemmaRmsNorm(const GemmaRmsNormRequest& request, const CudaLaunchContext& context);
bool GemmaRmsNormAvailable(const GemmaRmsNormRequest* probe, uint16_t compute_capability);
absl::Status QkRmsNorm(const QkRmsNormRequest& request, const CudaLaunchContext& context);
bool QkRmsNormAvailable(const QkRmsNormRequest* probe, uint16_t compute_capability);
absl::Status HadamardTransform(const HadamardTransformRequest& request,
                               const CudaLaunchContext& context);
bool HadamardTransformAvailable(const HadamardTransformRequest* probe,
                                uint16_t compute_capability);
absl::Status Argmax(const ArgmaxRequest& request, const CudaLaunchContext& context);
bool ArgmaxAvailable(const ArgmaxRequest* probe, uint16_t compute_capability);
absl::Status TopPRenorm(const TopPRenormRequest& request, const CudaLaunchContext& context);
bool TopPRenormAvailable(const TopPRenormRequest* probe, uint16_t compute_capability);
absl::Status TopKRenorm(const TopKRenormRequest& request, const CudaLaunchContext& context);
bool TopKRenormAvailable(const TopKRenormRequest* probe, uint16_t compute_capability);
absl::Status Fp8Quant(const Fp8QuantRequest& request, const CudaLaunchContext& context);
bool Fp8QuantAvailable(const Fp8QuantRequest* probe, uint16_t compute_capability);
absl::Status SoftmaxTopK(const SoftmaxTopKRequest& request, const CudaLaunchContext& context);
bool SoftmaxTopKAvailable(const SoftmaxTopKRequest* probe, uint16_t compute_capability);
absl::Status SigmoidBiasTopK(const SigmoidBiasTopKRequest& request,
                             const CudaLaunchContext& context);
bool SigmoidBiasTopKAvailable(const SigmoidBiasTopKRequest* probe, uint16_t compute_capability);
absl::Status AttnRes(const AttnResRequest& request, const CudaLaunchContext& context);
bool AttnResAvailable(const AttnResRequest* probe, uint16_t compute_capability);
absl::Status HcMix(const HcMixRequest& request, const CudaLaunchContext& context);
bool HcMixAvailable(const HcMixRequest* probe, uint16_t compute_capability);
absl::Status HcCombine(const HcCombineRequest& request, const CudaLaunchContext& context);
bool HcCombineAvailable(const HcCombineRequest* probe, uint16_t compute_capability);
absl::Status MhcPre(const MhcPreRequest& request, const CudaLaunchContext& context);
bool MhcPreAvailable(const MhcPreRequest* probe, uint16_t compute_capability);
absl::Status MhcPost(const MhcPostRequest& request, const CudaLaunchContext& context);
bool MhcPostAvailable(const MhcPostRequest* probe, uint16_t compute_capability);
absl::Status RingSconv(const RingSconvRequest& request, const CudaLaunchContext& context);
bool RingSconvAvailable(const RingSconvRequest* probe, uint16_t compute_capability);

}  // namespace inferx::kernels::cuda::launchers

#endif  // INFERX_KERNELS_CUDA_DISPATCH_OP_LAUNCHERS_H_
