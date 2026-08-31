#include "inferx/platform/cuda/cuda_stream.h"

#include <algorithm>
#include <utility>

#include "absl/status/status.h"
#include "inferx/platform/cuda/cuda_device.h"

namespace inferx::cuda {

CudaStream::CudaStream(const CudaApi* api, CudaHealth* health, DeviceId device, CudaStreamRole role,
                       int priority, cudaStream_t stream) noexcept
    : api_(api),
      health_(health),
      device_(device),
      role_(role),
      priority_(priority),
      stream_(stream) {}

absl::StatusOr<CudaStream> CudaStream::Create(DeviceId device, CudaStreamRole role,
                                              int requested_priority, const CudaApi& api,
                                              CudaHealth* health) {
  if (health != nullptr) {
    absl::Status status = health->CheckAcceptingWork();
    if (!status.ok()) return status;
  }
  if (role != CudaStreamRole::kCompute && role != CudaStreamRole::kTransfer) {
    return absl::InvalidArgumentError("cuda.stream.role: invalid stream role");
  }
  absl::StatusOr<CudaDeviceGuard> guard = CudaDeviceGuard::Create(device, api, health);
  if (!guard.ok()) return guard.status();
  int least_priority = 0;
  int greatest_priority = 0;
  cudaError_t error = api.device_get_stream_priority_range(&least_priority, &greatest_priority);
  if (error != cudaSuccess) {
    return CudaErrorStatus(error, "stream-priority-range", device, health);
  }
  const int actual_priority = std::clamp(requested_priority, greatest_priority, least_priority);
  cudaStream_t stream = nullptr;
  error = api.stream_create_with_priority(&stream, cudaStreamNonBlocking, actual_priority);
  if (error != cudaSuccess) {
    return CudaErrorStatus(error, "stream-create", device, health);
  }
  absl::Status restore = guard->Restore();
  if (!restore.ok()) {
    static_cast<void>(api.stream_destroy(stream));
    return restore;
  }
  return CudaStream(&api, health, device, role, actual_priority, stream);
}

CudaStream::~CudaStream() noexcept {
  if (stream_ != nullptr) {
    static_cast<void>(Close());
  }
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : api_(other.api_),
      health_(other.health_),
      device_(other.device_),
      role_(other.role_),
      priority_(other.priority_),
      stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  if (this != &other) {
    if (stream_ != nullptr) static_cast<void>(Close());
    api_ = other.api_;
    health_ = other.health_;
    device_ = other.device_;
    role_ = other.role_;
    priority_ = other.priority_;
    stream_ = std::exchange(other.stream_, nullptr);
  }
  return *this;
}

absl::Status CudaStream::Close() {
  if (stream_ == nullptr) return absl::OkStatus();
  cudaStream_t stream = std::exchange(stream_, nullptr);
  return CudaErrorStatus(api_->stream_destroy(stream), "stream-destroy", device_, health_);
}

}  // namespace inferx::cuda
