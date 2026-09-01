// CUDA error translation and sticky-context health.
#ifndef INFERX_PLATFORM_CUDA_CUDA_ERROR_H_
#define INFERX_PLATFORM_CUDA_CUDA_ERROR_H_

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "inferx/base/id.h"

namespace inferx::cuda {

enum class CudaHealthState : uint8_t {
  kHealthy,
  kPoisoned,
  kShuttingDown,
  kClosed,
};

class CudaHealth {
 public:
  [[nodiscard]] CudaHealthState state() const noexcept;
  [[nodiscard]] absl::Status CheckAcceptingWork() const;
  void Poison(absl::Status cause) noexcept;
  [[nodiscard]] absl::Status BeginShutdown();
  [[nodiscard]] absl::Status Close();
  [[nodiscard]] std::optional<absl::Status> poison_cause() const;

 private:
  mutable std::mutex mutex_;
  std::atomic<CudaHealthState> state_{CudaHealthState::kHealthy};
  std::optional<absl::Status> poison_cause_;
};

[[nodiscard]] bool IsCudaStickyFault(cudaError_t error) noexcept;
[[nodiscard]] absl::Status CudaErrorStatus(cudaError_t error, absl::string_view operation,
                                           DeviceId device, CudaHealth* health = nullptr);
[[nodiscard]] absl::Status CheckCuda(cudaError_t error, absl::string_view operation,
                                     DeviceId device, CudaHealth* health = nullptr);

}  // namespace inferx::cuda

#endif  // INFERX_PLATFORM_CUDA_CUDA_ERROR_H_
