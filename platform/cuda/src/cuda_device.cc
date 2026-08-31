#include "inferx/platform/cuda/cuda_device.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"
#include "inferx/platform/cuda/cuda_error.h"

namespace inferx::cuda {
namespace {

std::string SanitizeName(const char* raw) {
  std::string result(raw == nullptr ? "" : raw);
  for (char& value : result) {
    const unsigned char character = static_cast<unsigned char>(value);
    if (!std::isprint(character)) {
      value = '?';
    }
  }
  return result;
}

}  // namespace

absl::StatusOr<std::vector<CudaDeviceInfo>> DiscoverCudaDevices(const CudaApi& api) {
  int count = 0;
  cudaError_t error = api.get_device_count(&count);
  if (error == cudaErrorNoDevice || (error == cudaSuccess && count == 0)) {
    return absl::NotFoundError("cuda.discovery: no visible CUDA device");
  }
  if (error != cudaSuccess) {
    return CudaErrorStatus(error, "get-device-count", DeviceId(0));
  }
  int driver_version = 0;
  int runtime_version = 0;
  if (error = api.driver_get_version(&driver_version); error != cudaSuccess) {
    return CudaErrorStatus(error, "driver-version", DeviceId(0));
  }
  if (error = api.runtime_get_version(&runtime_version); error != cudaSuccess) {
    return CudaErrorStatus(error, "runtime-version", DeviceId(0));
  }
  try {
    std::vector<CudaDeviceInfo> result;
    result.reserve(static_cast<size_t>(count));
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      cudaDeviceProp properties{};
      error = api.get_device_properties(&properties, ordinal);
      if (error != cudaSuccess) {
        return CudaErrorStatus(error, "get-device-properties",
                               DeviceId(static_cast<uint32_t>(ordinal)));
      }
      result.push_back(CudaDeviceInfo{
          DeviceId(static_cast<uint32_t>(ordinal)), SanitizeName(properties.name), properties.major,
          properties.minor, ByteCount(static_cast<uint64_t>(properties.totalGlobalMem)),
          properties.multiProcessorCount, properties.warpSize, properties.maxThreadsPerBlock,
          properties.asyncEngineCount, properties.unifiedAddressing != 0,
          properties.canMapHostMemory != 0, properties.managedMemory != 0,
          properties.streamPrioritiesSupported != 0, properties.memoryPoolsSupported != 0,
          driver_version, runtime_version});
    }
    return result;
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("cuda.discovery: host result allocation failed");
  }
}

absl::Status ValidateCudaCapabilities(const CudaDeviceInfo& info, std::span<const int> accepted_sms,
                                      ByteCount device_reserve, ByteCount device_budget) {
  if (info.runtime_version < 12080) {
    return WithErrorReason(
        absl::UnimplementedError("cuda.capability: CUDA runtime 12.8 or newer is required"),
        ErrorReason::kUnsupportedCapability);
  }
  const int sm = info.compute_major * 10 + info.compute_minor;
  if (std::find(accepted_sms.begin(), accepted_sms.end(), sm) == accepted_sms.end()) {
    return WithErrorReason(
        absl::UnimplementedError("cuda.capability: compute capability is not accepted"),
        ErrorReason::kUnsupportedCapability);
  }
  if (!info.unified_addressing) {
    return WithErrorReason(
        absl::UnimplementedError("cuda.capability: unified addressing is required"),
        ErrorReason::kUnsupportedCapability);
  }
  if (device_reserve.value() > info.total_memory.value()) {
    return absl::InvalidArgumentError("cuda.device_reserve_bytes: reserve exceeds total memory");
  }
  const uint64_t allocatable = info.total_memory.value() - device_reserve.value();
  if (device_budget.value() != 0 && device_budget.value() > allocatable) {
    return absl::InvalidArgumentError(
        "cuda.device_budget_bytes: budget plus reserve exceeds total memory");
  }
  absl::StatusOr<size_t> narrowed = CheckedNarrow<size_t>(
      device_budget.value() == 0 ? allocatable : device_budget.value(), "cuda.device_budget_bytes");
  if (!narrowed.ok()) {
    return narrowed.status();
  }
  return absl::OkStatus();
}

CudaDeviceGuard::CudaDeviceGuard(const CudaApi* api, DeviceId requested, int previous, bool changed,
                                 CudaHealth* health) noexcept
    : api_(api), requested_(requested), previous_(previous), changed_(changed), health_(health) {}

absl::StatusOr<CudaDeviceGuard> CudaDeviceGuard::Create(DeviceId requested, const CudaApi& api,
                                                        CudaHealth* health) {
  if (requested.value() > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return absl::InvalidArgumentError("cuda.device: ordinal cannot narrow to int");
  }
  int previous = 0;
  cudaError_t error = api.get_device(&previous);
  if (error != cudaSuccess) {
    return CudaErrorStatus(error, "get-device", requested, health);
  }
  const int requested_int = static_cast<int>(requested.value());
  const bool changed = previous != requested_int;
  if (changed) {
    error = api.set_device(requested_int);
    if (error != cudaSuccess) {
      return CudaErrorStatus(error, "set-device", requested, health);
    }
  }
  return CudaDeviceGuard(&api, requested, previous, changed, health);
}

CudaDeviceGuard::~CudaDeviceGuard() noexcept {
  if (api_ != nullptr) {
    static_cast<void>(Restore());
  }
}

CudaDeviceGuard::CudaDeviceGuard(CudaDeviceGuard&& other) noexcept
    : api_(std::exchange(other.api_, nullptr)),
      requested_(other.requested_),
      previous_(other.previous_),
      changed_(other.changed_),
      health_(other.health_) {}

CudaDeviceGuard& CudaDeviceGuard::operator=(CudaDeviceGuard&& other) noexcept {
  if (this != &other) {
    if (api_ != nullptr) static_cast<void>(Restore());
    api_ = std::exchange(other.api_, nullptr);
    requested_ = other.requested_;
    previous_ = other.previous_;
    changed_ = other.changed_;
    health_ = other.health_;
  }
  return *this;
}

absl::Status CudaDeviceGuard::Restore() {
  if (api_ == nullptr) {
    return absl::OkStatus();
  }
  const CudaApi* api = std::exchange(api_, nullptr);
  if (!changed_) {
    return absl::OkStatus();
  }
  absl::Status status =
      CudaErrorStatus(api->set_device(previous_), "restore-device", requested_, health_);
  if (!status.ok() && health_ != nullptr) {
    health_->Poison(status);
  }
  return status;
}

}  // namespace inferx::cuda
