#include "inferx/platform/cuda/cuda_error.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <mutex>
#include <optional>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "inferx/base/status.h"
#include "inferx/platform/cuda/cuda_api.h"

namespace inferx::cuda {

CudaHealthState CudaHealth::state() const noexcept {
  return state_.load(std::memory_order_acquire);
}

absl::Status CudaHealth::CheckAcceptingWork() const {
  CudaHealthState state = state_.load(std::memory_order_acquire);
  if (state == CudaHealthState::kHealthy) return absl::OkStatus();
  std::lock_guard<std::mutex> lock(mutex_);
  state = state_.load(std::memory_order_relaxed);
  switch (state) {
    case CudaHealthState::kHealthy:
      return absl::OkStatus();
    case CudaHealthState::kPoisoned:
      return poison_cause_.value_or(absl::UnavailableError("cuda.health: context is poisoned"));
    case CudaHealthState::kShuttingDown:
      return absl::FailedPreconditionError("cuda.health: context is shutting down");
    case CudaHealthState::kClosed:
      return absl::FailedPreconditionError("cuda.health: context is closed");
  }
  return absl::InternalError("cuda.health: invalid health state");
}

void CudaHealth::Poison(absl::Status cause) noexcept {
  if (cause.ok()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  CudaHealthState state = state_.load(std::memory_order_relaxed);
  if (!poison_cause_.has_value() && state != CudaHealthState::kClosed) {
    poison_cause_ = std::move(cause);
  }
  if (state == CudaHealthState::kHealthy) {
    state_.store(CudaHealthState::kPoisoned, std::memory_order_release);
  }
}

absl::Status CudaHealth::BeginShutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  const CudaHealthState state = state_.load(std::memory_order_relaxed);
  if (state == CudaHealthState::kClosed) {
    return absl::OkStatus();
  }
  if (state == CudaHealthState::kShuttingDown) {
    return absl::OkStatus();
  }
  state_.store(CudaHealthState::kShuttingDown, std::memory_order_release);
  return absl::OkStatus();
}

absl::Status CudaHealth::Close() {
  std::lock_guard<std::mutex> lock(mutex_);
  state_.store(CudaHealthState::kClosed, std::memory_order_release);
  return absl::OkStatus();
}

std::optional<absl::Status> CudaHealth::poison_cause() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return poison_cause_;
}

bool IsCudaStickyFault(cudaError_t error) noexcept {
  switch (error) {
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
    case cudaErrorLaunchTimeout:
    case cudaErrorAssert:
    case cudaErrorContextIsDestroyed:
      return true;
    default:
      return false;
  }
}

absl::Status CudaErrorStatus(cudaError_t error, absl::string_view operation, DeviceId device,
                             CudaHealth* health) {
  if (error == cudaSuccess) {
    return absl::OkStatus();
  }
  absl::StatusCode code = absl::StatusCode::kInternal;
  ErrorReason reason = ErrorReason::kCudaApiFailure;
  switch (error) {
    case cudaErrorInvalidDevice:
      code = absl::StatusCode::kInvalidArgument;
      reason = ErrorReason::kCudaInvalidDevice;
      break;
    case cudaErrorMemoryAllocation:
      code = absl::StatusCode::kResourceExhausted;
      reason = ErrorReason::kCudaOutOfMemory;
      break;
    case cudaErrorInvalidConfiguration:
      code = absl::StatusCode::kInvalidArgument;
      reason = ErrorReason::kCudaLaunchRejected;
      break;
    case cudaErrorLaunchOutOfResources:
      code = absl::StatusCode::kResourceExhausted;
      reason = ErrorReason::kCudaLaunchRejected;
      break;
    case cudaErrorInvalidResourceHandle:
      code = absl::StatusCode::kInternal;
      reason = ErrorReason::kInvariantViolation;
      break;
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
    case cudaErrorLaunchTimeout:
    case cudaErrorAssert:
      code = absl::StatusCode::kUnavailable;
      reason = ErrorReason::kCudaAsyncFault;
      break;
    case cudaErrorContextIsDestroyed:
      code = absl::StatusCode::kUnavailable;
      reason = ErrorReason::kCudaDeviceLost;
      break;
    case cudaErrorInsufficientDriver:
      code = absl::StatusCode::kFailedPrecondition;
      reason = ErrorReason::kUnsupportedCapability;
      break;
    default:
      break;
  }
  absl::Status status = WithErrorReason(
      absl::Status(
          code, absl::StrCat("cuda.", operation, ": ", CudaApi::Production().get_error_name(error),
                             " (", static_cast<int>(error), ") on device ", device.value(), ": ",
                             CudaApi::Production().get_error_string(error))),
      reason);
  if (health != nullptr && (IsCudaStickyFault(error) || error == cudaErrorInvalidResourceHandle)) {
    health->Poison(status);
  }
  return status;
}

absl::Status CheckCuda(cudaError_t error, absl::string_view operation, DeviceId device,
                       CudaHealth* health) {
  return CudaErrorStatus(error, operation, device, health);
}

}  // namespace inferx::cuda
