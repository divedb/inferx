#ifndef INFERX_KERNELS_MEMORY_H_
#define INFERX_KERNELS_MEMORY_H_

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace inferx::kernels {

/// \brief A non-owning view of a contiguous block of device memory.
struct DeviceBytes {
  const void* data = nullptr;
  uint64_t size = 0;
};

/// \brief A non-owning view of a contiguous block of mutable device memory.
struct MutableDeviceBytes {
  void* data = nullptr;
  uint64_t size = 0;
};

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_MEMORY_H_
