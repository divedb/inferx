// Build-only kernels-layer access to validated buffer addresses. This
// accessor moved from the retired platform layer (ADR 0031 retirement
// addendum): raw-address resolution is needed by every device backend, so
// it lives with the unified kernels surface.
#ifndef INFERX_KERNELS_CUDA_BUFFER_ACCESS_H_
#define INFERX_KERNELS_CUDA_BUFFER_ACCESS_H_

#include "inferx/tensor/buffer.h"

namespace inferx::kernels {

class BufferAccess {
 public:
  [[nodiscard]] static const void* Address(const BufferView& view) noexcept {
    return view.address_;
  }
  [[nodiscard]] static void* Address(const MutableBufferView& view) noexcept {
    return view.address_;
  }
};

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_CUDA_BUFFER_ACCESS_H_
