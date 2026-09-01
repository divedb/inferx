// Device-owned nonblocking CUDA stream.
#ifndef INFERX_PLATFORM_CUDA_CUDA_STREAM_H_
#define INFERX_PLATFORM_CUDA_CUDA_STREAM_H_

#include <cuda_runtime_api.h>

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/platform/cuda/cuda_api.h"
#include "inferx/platform/cuda/cuda_error.h"

namespace inferx::cuda {

enum class CudaStreamRole : uint8_t { kCompute, kTransfer };

class CudaStream {
 public:
  [[nodiscard]] static absl::StatusOr<CudaStream> Create(DeviceId device, CudaStreamRole role,
                                                         int requested_priority,
                                                         const CudaApi& api = CudaApi::Production(),
                                                         CudaHealth* health = nullptr);
  ~CudaStream() noexcept;
  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  [[nodiscard]] DeviceId device() const noexcept { return device_; }
  [[nodiscard]] CudaStreamRole role() const noexcept { return role_; }
  [[nodiscard]] int priority() const noexcept { return priority_; }
  [[nodiscard]] cudaStream_t handle() const noexcept { return stream_; }
  [[nodiscard]] absl::Status Close();

 private:
  CudaStream(const CudaApi* api, CudaHealth* health, DeviceId device, CudaStreamRole role,
             int priority, cudaStream_t stream) noexcept;

  const CudaApi* api_ = nullptr;
  CudaHealth* health_ = nullptr;
  DeviceId device_ = DeviceId(0);
  CudaStreamRole role_ = CudaStreamRole::kCompute;
  int priority_ = 0;
  cudaStream_t stream_ = nullptr;
};

}  // namespace inferx::cuda

#endif  // INFERX_PLATFORM_CUDA_CUDA_STREAM_H_
