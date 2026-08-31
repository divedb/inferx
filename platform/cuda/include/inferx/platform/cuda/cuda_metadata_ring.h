// Paired pinned/device metadata slots retained through consumer completion.
#ifndef INFERX_PLATFORM_CUDA_CUDA_METADATA_RING_H_
#define INFERX_PLATFORM_CUDA_CUDA_METADATA_RING_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/platform/cuda/cuda_allocator.h"
#include "inferx/platform/cuda/cuda_api.h"
#include "inferx/platform/cuda/cuda_event_pool.h"
#include "inferx/tensor/buffer.h"

namespace inferx::cuda {

struct MetadataToken {
  PoolSlotId slot;
  PoolGeneration generation;
  friend constexpr bool operator==(const MetadataToken&, const MetadataToken&) = default;
};

class MetadataRingState;

class CudaMetadataLease {
 public:
  CudaMetadataLease() noexcept = default;
  ~CudaMetadataLease() noexcept;
  CudaMetadataLease(CudaMetadataLease&& other) noexcept;
  CudaMetadataLease& operator=(CudaMetadataLease&& other) noexcept;
  CudaMetadataLease(const CudaMetadataLease&) = delete;
  CudaMetadataLease& operator=(const CudaMetadataLease&) = delete;

  [[nodiscard]] MetadataToken token() const noexcept { return token_; }
  [[nodiscard]] absl::StatusOr<MutableBufferView> host_view();
  [[nodiscard]] absl::StatusOr<BufferView> SealAndUpload(CudaStream& transfer, CudaStream& compute);
  [[nodiscard]] absl::StatusOr<BufferView> device_view() const;
  // Caller invokes this only after the consumer CompletionFence has been
  // acknowledged; the retained upload event is then known complete.
  [[nodiscard]] absl::Status Release();

 private:
  friend class CudaMetadataRing;
  CudaMetadataLease(std::shared_ptr<MetadataRingState> state, MetadataToken token) noexcept;
  std::shared_ptr<MetadataRingState> state_;
  MetadataToken token_{PoolSlotId(0), PoolGeneration(0)};
  bool sealed_ = false;
};

class CudaMetadataRing {
 public:
  [[nodiscard]] static absl::StatusOr<CudaMetadataRing> Create(
      uint32_t slots, ByteCount slot_bytes, CudaPinnedAllocator& pinned_allocator,
      CudaDeviceAllocator& device_allocator, CudaEventPool& event_pool, DeviceId device,
      const CudaApi& api = CudaApi::Production());
  CudaMetadataRing(CudaMetadataRing&&) noexcept = default;
  CudaMetadataRing& operator=(CudaMetadataRing&&) noexcept = default;
  CudaMetadataRing(const CudaMetadataRing&) = delete;
  CudaMetadataRing& operator=(const CudaMetadataRing&) = delete;
  ~CudaMetadataRing() noexcept;

  [[nodiscard]] absl::StatusOr<CudaMetadataLease> Acquire();
  [[nodiscard]] absl::Status ValidateInvariants() const;
  [[nodiscard]] absl::Status Close();

 private:
  explicit CudaMetadataRing(std::shared_ptr<MetadataRingState> state) noexcept;
  std::shared_ptr<MetadataRingState> state_;
};

}  // namespace inferx::cuda

#endif  // INFERX_PLATFORM_CUDA_CUDA_METADATA_RING_H_
