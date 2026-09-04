#ifndef INFERX_KERNELS_MEMORY_H_
#define INFERX_KERNELS_MEMORY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "absl/status/status.h"
#include "kernels/cuda/memory/context.h"

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

/// \brief A trivially-copyable value repeated across a destination range.
///
/// The public bound keeps the pattern cheap to pass by value to accelerator
/// backends. A destination byte count must be an exact multiple of `size`.
struct FillPattern {
  static constexpr uint8_t kMaxSize = 16;

  std::array<std::byte, kMaxSize> bytes{};
  uint8_t size = 0;

  template <typename T>
  [[nodiscard]] static FillPattern FromValue(const T& value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>,
                  "device fill values must be trivially copyable");
    static_assert(sizeof(T) <= kMaxSize,
                  "device fill values may contain at most 16 bytes");

    FillPattern pattern;
    std::memcpy(pattern.bytes.data(), &value, sizeof(T));
    pattern.size = static_cast<uint8_t>(sizeof(T));
    return pattern;
  }
};

/// \brief Asynchronously fills every byte in `destination` with `value`.
absl::Status FillBytesAsync(ExecutionContext context,
                            MutableDeviceBytes destination, std::byte value);

/// \brief Asynchronously repeats `pattern` across `destination`.
absl::Status FillAsync(ExecutionContext context, MutableDeviceBytes destination,
                       FillPattern pattern);

/// \brief Asynchronously zeroes `destination`.
absl::Status ZeroAsync(ExecutionContext context,
                       MutableDeviceBytes destination);

/// \brief Asynchronously performs an exact-size device-to-device copy.
///
/// Identical ranges are a no-op. Partially overlapping ranges are rejected.
absl::Status CopyAsync(ExecutionContext context, MutableDeviceBytes destination,
                       DeviceBytes source);

/// \brief Typed convenience overload for `FillAsync`.
template <typename T>
absl::Status FillAsync(ExecutionContext context, T* destination,
                       uint64_t element_count, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>,
                "device fill values must be trivially copyable");
  if (element_count > std::numeric_limits<uint64_t>::max() / sizeof(T)) {
    return absl::InvalidArgumentError("typed fill byte size overflows uint64");
  }
  return FillAsync(context,
                   MutableDeviceBytes{destination, element_count * sizeof(T)},
                   FillPattern::FromValue(value));
}

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_MEMORY_H_
