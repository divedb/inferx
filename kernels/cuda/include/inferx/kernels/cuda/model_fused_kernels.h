// Model-fused kernel shims: attn_res, hyperconnection, mhc, ring sconv.
#ifndef INFERX_KERNELS_CUDA_MODEL_FUSED_KERNELS_H_
#define INFERX_KERNELS_CUDA_MODEL_FUSED_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "inferx/kernels/cuda/storage_type.h"

namespace inferx::kernels::cuda::attnres {

// layer_residual [tokens, hidden] in/out; block_residual [blocks, tokens,
// hidden]; weights [hidden]; optional fused output norm (null to skip).
cudaError_t LaunchAttnRes(const void* block_residual, void* layer_residual, const void* res_weight,
                          const void* rms_weight, const void* out_norm_weight, uint64_t tokens,
                          uint64_t hidden, uint64_t blocks, float epsilon, float out_norm_epsilon,
                          StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::attnres

namespace inferx::kernels::cuda::hc {

// Full gated_residual_mix sequence. workspace layout (FP32, bytes):
// [0, tokens*rank) projected, [tokens*rank, tokens*rank + tokens*wide) gate.
uint64_t HcMixWorkspaceBytes(uint64_t tokens, uint32_t rank, uint32_t hc_count, uint32_t hidden);
cudaError_t LaunchHcMix(const void* normalized, const void* projection_weight,
                        const void* up_weight, void* mixed, void* inject_logits, void* workspace,
                        uint64_t tokens, uint32_t rank, uint32_t hc_count, uint32_t hidden,
                        float projection_scale, StorageType dtype, cudaStream_t stream);
cudaError_t LaunchHcCombine(const void* block_output, void* residual, const void* inject_logits,
                            uint64_t tokens, uint32_t hc_count, uint32_t hidden, StorageType dtype,
                            cudaStream_t stream);

}  // namespace inferx::kernels::cuda::hc

namespace inferx::kernels::cuda::mhc {

// mhc_pre: workspace must hold tokens * (2m + m*m) FP32 mix values.
uint64_t MhcPreWorkspaceBytes(uint64_t tokens, uint32_t streams);
cudaError_t LaunchMhcPre(const void* residual, const void* fn, const void* hc_scale,
                         const void* hc_base, void* mix_workspace, void* layer_input, void* post,
                         void* comb, uint64_t tokens, uint32_t streams, uint32_t hidden,
                         float rms_eps, float hc_eps, uint32_t sinkhorn_iters, StorageType dtype,
                         cudaStream_t stream);
cudaError_t LaunchMhcPost(const void* hidden_states, void* residual, const void* post,
                          const void* comb, uint64_t tokens, uint32_t streams, uint32_t hidden,
                          StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::mhc

namespace inferx::kernels::cuda::conv {

// Depthwise causal FIR with ring cache. seq_prefix [batch+1] device int32
// token offsets (prefix sums); weight [width, window]; conv_cache
// [slots, ring, width] updated in place.
cudaError_t LaunchRingSconv(const void* x, const void* weight, void* conv_cache,
                            const int32_t* seq_prefix, const int32_t* cache_indices, void* output,
                            uint64_t tokens, uint64_t width, uint32_t window, uint64_t ring,
                            uint64_t batch, StorageType dtype, cudaStream_t stream);

}  // namespace inferx::kernels::cuda::conv

#endif  // INFERX_KERNELS_CUDA_MODEL_FUSED_KERNELS_H_
