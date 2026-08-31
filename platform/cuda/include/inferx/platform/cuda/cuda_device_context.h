// Worker-owned CUDA context and transactional M2 pool ownership.
#ifndef INFERX_PLATFORM_CUDA_CUDA_DEVICE_CONTEXT_H_
#define INFERX_PLATFORM_CUDA_CUDA_DEVICE_CONTEXT_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/platform/cuda/cuda_allocator.h"
#include "inferx/platform/cuda/cuda_api.h"
#include "inferx/platform/cuda/cuda_device.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/cuda_event_pool.h"
#include "inferx/platform/cuda/cuda_metadata_ring.h"
#include "inferx/platform/cuda/cuda_stream.h"
#include "inferx/runtime/buffer_pool.h"
#include "inferx/runtime/memory_tracker.h"
#include "inferx/runtime/workspace.h"

namespace inferx::cuda {

struct CudaContextConfig {
  DeviceId device = DeviceId(0);
  int accepted_sm = 89;
  ByteCount device_reserve = ByteCount(512ULL * 1024 * 1024);
  ByteCount device_budget = ByteCount(0);
  ByteCount pinned_budget = ByteCount(256ULL * 1024 * 1024);
  uint32_t event_pool_slots = 1024;
  uint32_t metadata_ring_slots = 3;
  ByteCount metadata_slot_bytes = ByteCount(1024 * 1024);
  uint32_t staging_pool_slots = 4;
  ByteCount staging_slot_bytes = ByteCount(4ULL * 1024 * 1024);
  uint32_t workspace_slots = 4;
  ByteCount workspace_bytes_per_slot = ByteCount(16ULL * 1024 * 1024);
  bool enable_transfer_stream = true;
};

class CudaDeviceContext {
 public:
  [[nodiscard]] static absl::StatusOr<std::unique_ptr<CudaDeviceContext>> Create(
      const CudaContextConfig& config, MemoryTracker& tracker,
      const CudaApi& api = CudaApi::Production());
  ~CudaDeviceContext() noexcept;
  CudaDeviceContext(const CudaDeviceContext&) = delete;
  CudaDeviceContext& operator=(const CudaDeviceContext&) = delete;

  [[nodiscard]] const CudaDeviceInfo& info() const noexcept;
  [[nodiscard]] CudaHealth& health() noexcept;
  [[nodiscard]] CudaStream& compute_stream() noexcept;
  [[nodiscard]] CudaStream& transfer_stream() noexcept;
  [[nodiscard]] CudaEventPool& event_pool() noexcept;
  [[nodiscard]] CudaDeviceAllocator& device_allocator() noexcept;
  [[nodiscard]] CudaPinnedAllocator& pinned_allocator() noexcept;
  [[nodiscard]] FixedBufferPool& staging_pool() noexcept;
  [[nodiscard]] WorkspaceArenaPool& workspace_pool() noexcept;
  [[nodiscard]] CudaMetadataRing& metadata_ring() noexcept;
  [[nodiscard]] absl::Status Shutdown(Deadline deadline);

 private:
  struct Impl;
  explicit CudaDeviceContext(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::cuda

#endif  // INFERX_PLATFORM_CUDA_CUDA_DEVICE_CONTEXT_H_
