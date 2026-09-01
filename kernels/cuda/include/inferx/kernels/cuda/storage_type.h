// Closed storage-type vocabulary shared by the per-category CUDA launch
// shims (ADR 0030 policy: CUDA shims carry no Abseil types).
#ifndef INFERX_KERNELS_CUDA_STORAGE_TYPE_H_
#define INFERX_KERNELS_CUDA_STORAGE_TYPE_H_

#include <cstdint>

namespace inferx::kernels::cuda {

enum class StorageType : uint8_t { kFloat32, kFloat16, kBFloat16 };

}  // namespace inferx::kernels::cuda

#endif  // INFERX_KERNELS_CUDA_STORAGE_TYPE_H_
