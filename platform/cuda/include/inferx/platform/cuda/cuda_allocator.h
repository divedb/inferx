// Startup/large-ownership CUDA device and pinned-host allocators.
#ifndef INFERX_PLATFORM_CUDA_CUDA_ALLOCATOR_H_
#define INFERX_PLATFORM_CUDA_CUDA_ALLOCATOR_H_

#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/platform/cuda/cuda_api.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/runtime/memory_tracker.h"
#include "inferx/tensor/allocator.h"

namespace inferx::cuda {

class CudaDeviceAllocator final : public Allocator {
 public:
  CudaDeviceAllocator(DeviceId device, MemoryTracker& tracker, CudaHealth& health,
                      const CudaApi& api = CudaApi::Production()) noexcept;
  ~CudaDeviceAllocator() override = default;
  CudaDeviceAllocator(CudaDeviceAllocator&&) noexcept;
  CudaDeviceAllocator& operator=(CudaDeviceAllocator&&) noexcept;
  CudaDeviceAllocator(const CudaDeviceAllocator&) = delete;
  CudaDeviceAllocator& operator=(const CudaDeviceAllocator&) = delete;

  [[nodiscard]] absl::StatusOr<Buffer> Allocate(const AllocationRequest& request) override;

 private:
  DeviceId device_ = DeviceId(0);
  MemoryTracker* tracker_ = nullptr;
  CudaHealth* health_ = nullptr;
  const CudaApi* api_ = nullptr;
  bool active_ = true;
};

class CudaPinnedAllocator final : public Allocator {
 public:
  CudaPinnedAllocator(DeviceId owning_device, MemoryTracker& tracker, CudaHealth& health,
                      const CudaApi& api = CudaApi::Production()) noexcept;
  ~CudaPinnedAllocator() override = default;
  CudaPinnedAllocator(CudaPinnedAllocator&&) noexcept;
  CudaPinnedAllocator& operator=(CudaPinnedAllocator&&) noexcept;
  CudaPinnedAllocator(const CudaPinnedAllocator&) = delete;
  CudaPinnedAllocator& operator=(const CudaPinnedAllocator&) = delete;

  [[nodiscard]] absl::StatusOr<Buffer> Allocate(const AllocationRequest& request) override;

 private:
  DeviceId owning_device_ = DeviceId(0);
  MemoryTracker* tracker_ = nullptr;
  CudaHealth* health_ = nullptr;
  const CudaApi* api_ = nullptr;
  bool active_ = true;
};

}  // namespace inferx::cuda

#endif  // INFERX_PLATFORM_CUDA_CUDA_ALLOCATOR_H_
