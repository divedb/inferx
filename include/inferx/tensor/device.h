// Device and memory-address-space vocabulary for InferX tensors.
#ifndef INFERX_TENSOR_DEVICE_H_
#define INFERX_TENSOR_DEVICE_H_

#include <compare>
#include <cstdint>

#include "absl/status/status.h"
#include "inferx/base/id.h"

namespace inferx {

enum class DeviceKind : uint8_t { kHost = 0, kCuda = 1 };

enum class MemoryKind : uint8_t {
  kHost = 0,
  kPinnedHost = 1,
  kDevice = 2,
  kManaged = 3,
};

struct Device {
  DeviceKind kind;
  DeviceId ordinal;

  [[nodiscard]] static constexpr Device Host() noexcept {
    return Device{DeviceKind::kHost, DeviceId(0)};
  }
  [[nodiscard]] static constexpr Device Cuda(DeviceId ordinal) noexcept {
    return Device{DeviceKind::kCuda, ordinal};
  }
  [[nodiscard]] absl::Status Validate() const;

  friend constexpr bool operator==(const Device&, const Device&) = default;
  friend constexpr auto operator<=>(const Device&, const Device&) = default;
};

[[nodiscard]] absl::Status ValidateMemoryKind(Device device, MemoryKind memory_kind);
[[nodiscard]] constexpr bool IsHostAddressable(MemoryKind kind) noexcept {
  return kind == MemoryKind::kHost || kind == MemoryKind::kPinnedHost;
}

}  // namespace inferx

#endif  // INFERX_TENSOR_DEVICE_H_
