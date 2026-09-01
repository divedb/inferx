// Value-only CUDA discovery and capability validation.
#ifndef INFERX_PLATFORM_CUDA_CUDA_DEVICE_H_
#define INFERX_PLATFORM_CUDA_CUDA_DEVICE_H_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/platform/cuda/cuda_api.h"
#include "inferx/platform/cuda/cuda_error.h"

namespace inferx::cuda {

struct CudaDeviceInfo {
  DeviceId ordinal;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  ByteCount total_memory = ByteCount(0);
  int multiprocessor_count = 0;
  int warp_size = 0;
  int max_threads_per_block = 0;
  int async_engine_count = 0;
  bool unified_addressing = false;
  bool can_map_host_memory = false;
  bool managed_memory = false;
  bool stream_priorities = false;
  bool memory_pools = false;
  int driver_version = 0;
  int runtime_version = 0;
};

[[nodiscard]] absl::StatusOr<std::vector<CudaDeviceInfo>> DiscoverCudaDevices(
    const CudaApi& api = CudaApi::Production());
[[nodiscard]] absl::Status ValidateCudaCapabilities(const CudaDeviceInfo& info,
                                                    std::span<const int> accepted_sms,
                                                    ByteCount device_reserve,
                                                    ByteCount device_budget);

class CudaDeviceGuard {
 public:
  [[nodiscard]] static absl::StatusOr<CudaDeviceGuard> Create(
      DeviceId requested, const CudaApi& api = CudaApi::Production(), CudaHealth* health = nullptr);
  ~CudaDeviceGuard() noexcept;
  CudaDeviceGuard(CudaDeviceGuard&& other) noexcept;
  CudaDeviceGuard& operator=(CudaDeviceGuard&& other) noexcept;
  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

  [[nodiscard]] absl::Status Restore();

 private:
  CudaDeviceGuard(const CudaApi* api, DeviceId requested, int previous, bool changed,
                  CudaHealth* health) noexcept;
  const CudaApi* api_ = nullptr;
  DeviceId requested_ = DeviceId(0);
  int previous_ = 0;
  bool changed_ = false;
  CudaHealth* health_ = nullptr;
};

}  // namespace inferx::cuda

#endif  // INFERX_PLATFORM_CUDA_CUDA_DEVICE_H_
