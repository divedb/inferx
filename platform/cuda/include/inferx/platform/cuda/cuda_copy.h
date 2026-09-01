// Strictly asynchronous CUDA copy/memset adapters.
#ifndef INFERX_PLATFORM_CUDA_CUDA_COPY_H_
#define INFERX_PLATFORM_CUDA_CUDA_COPY_H_

#include <cstdint>

#include "absl/status/status.h"
#include "inferx/base/token.h"
#include "inferx/platform/cuda/cuda_api.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/cuda_stream.h"
#include "inferx/tensor/buffer.h"

namespace inferx::cuda {

struct CopyRequest {
  BufferView source;
  MutableBufferView destination;
  ByteCount bytes;
};

[[nodiscard]] absl::Status CopyAsync(const CopyRequest& request, CudaStream& stream,
                                     const CudaApi& api = CudaApi::Production(),
                                     CudaHealth* health = nullptr);
[[nodiscard]] absl::Status MemsetAsync(MutableBufferView destination, uint8_t value,
                                       CudaStream& stream,
                                       const CudaApi& api = CudaApi::Production(),
                                       CudaHealth* health = nullptr);

}  // namespace inferx::cuda

#endif  // INFERX_PLATFORM_CUDA_CUDA_COPY_H_
