#include "inferx/tensor/device.h"

#include "absl/status/status.h"

namespace inferx {

absl::Status Device::Validate() const {
  switch (kind) {
    case DeviceKind::kHost:
      if (ordinal.value() != 0) {
        return absl::InvalidArgumentError("device.ordinal: host ordinal must be zero");
      }
      return absl::OkStatus();
    case DeviceKind::kCuda:
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError("device.kind: invalid device kind");
}

absl::Status ValidateMemoryKind(Device device, MemoryKind memory_kind) {
  absl::Status status = device.Validate();
  if (!status.ok()) {
    return status;
  }
  switch (memory_kind) {
    case MemoryKind::kHost:
      if (device.kind != DeviceKind::kHost) {
        return absl::InvalidArgumentError("allocation.memory_kind: host memory needs host device");
      }
      return absl::OkStatus();
    case MemoryKind::kPinnedHost:
      if (device.kind != DeviceKind::kHost) {
        return absl::InvalidArgumentError(
            "allocation.memory_kind: pinned memory reports host address space");
      }
      return absl::OkStatus();
    case MemoryKind::kDevice:
      if (device.kind != DeviceKind::kCuda) {
        return absl::InvalidArgumentError(
            "allocation.memory_kind: device memory needs a CUDA device");
      }
      return absl::OkStatus();
    case MemoryKind::kManaged:
      return absl::UnimplementedError("allocation.memory_kind: managed memory is unsupported");
  }
  return absl::InvalidArgumentError("allocation.memory_kind: invalid memory kind");
}

}  // namespace inferx
