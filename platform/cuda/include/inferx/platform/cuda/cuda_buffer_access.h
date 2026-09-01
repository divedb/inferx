// Build-only CUDA access to validated buffer addresses.
#ifndef INFERX_PLATFORM_CUDA_CUDA_BUFFER_ACCESS_H_
#define INFERX_PLATFORM_CUDA_CUDA_BUFFER_ACCESS_H_

#include "inferx/tensor/buffer.h"

namespace inferx::cuda {

class BufferAccess {
 public:
  [[nodiscard]] static const void* Address(const BufferView& view) noexcept {
    return view.address_;
  }
  [[nodiscard]] static void* Address(const MutableBufferView& view) noexcept {
    return view.address_;
  }
};

}  // namespace inferx::cuda

#endif  // INFERX_PLATFORM_CUDA_CUDA_BUFFER_ACCESS_H_
