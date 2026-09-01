#include "inferx/platform/cuda/cuda_allocator.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/status.h"
#include "inferx/platform/cuda/cuda_device.h"
#include "inferx/tensor/buffer.h"

namespace inferx::cuda {
namespace {

enum class AllocationFlavor : uint8_t { kDevice, kPinned };

ByteCount PointerAlignment(void* address) {
  const uintptr_t numeric = reinterpret_cast<uintptr_t>(address);
  if (numeric == 0) return ByteCount(1);
  return ByteCount(uint64_t{1} << std::countr_zero(numeric));
}

absl::Status AnnotateCleanup(absl::Status primary, const absl::Status& cleanup) {
  if (cleanup.ok()) return primary;
  absl::Status annotated(
      primary.code(), absl::StrCat(primary.message(), "; cleanup failure: ", cleanup.ToString()));
  primary.ForEachPayload([&annotated](absl::string_view url, const absl::Cord& payload) {
    annotated.SetPayload(url, payload);
  });
  return annotated;
}

cudaError_t FreeRawAllocation(void* address, AllocationFlavor flavor, const CudaApi& api) {
  return flavor == AllocationFlavor::kDevice ? api.free_device(address) : api.free_host(address);
}

absl::Status CleanupUncommittedAllocation(void* address, AllocationFlavor flavor,
                                          DeviceId cuda_device, MemoryReservation& reservation,
                                          CudaHealth& health, const CudaApi& api,
                                          absl::Status primary) {
  const cudaError_t error = FreeRawAllocation(address, flavor, api);
  if (error == cudaSuccess) return primary;

  absl::Status cleanup = CudaErrorStatus(
      error, flavor == AllocationFlavor::kDevice ? "free" : "free-host", cuda_device, &health);
  health.Poison(cleanup);

  // The runtime still owns bytes that failed to free. Convert their reservation
  // to an explicit leaked charge so the tracker never pretends capacity returned.
  absl::StatusOr<AllocationCharge> charge = reservation.Commit();
  if (!charge.ok()) {
    cleanup = absl::Status(
        cleanup.code(),
        absl::StrCat(cleanup.message(), "; leak-accounting failure: ", charge.status().ToString()));
  } else {
    const absl::Status leaked = charge->MarkLeaked();
    if (!leaked.ok()) {
      cleanup = absl::Status(
          cleanup.code(),
          absl::StrCat(cleanup.message(), "; leak-accounting failure: ", leaked.ToString()));
    }
  }
  return AnnotateCleanup(std::move(primary), cleanup);
}

class CudaAllocationDomain final : public AllocationDomain {
 public:
  CudaAllocationDomain(const CudaApi* api, CudaHealth* health, DeviceId cuda_device,
                       AllocationId id, AllocationFlavor flavor)
      : api_(api), health_(health), cuda_device_(cuda_device), id_(id), flavor_(flavor) {}

  void AdoptCharge(AllocationCharge charge) noexcept { charge_ = std::move(charge); }

  absl::Status Release(void* address, ByteCount, ByteCount, AllocationId id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_ || id != id_) {
      return absl::FailedPreconditionError(
          "cuda.allocation.release: duplicate or foreign allocation ID");
    }
    const cudaError_t error = flavor_ == AllocationFlavor::kDevice ? api_->free_device(address)
                                                                   : api_->free_host(address);
    if (error != cudaSuccess) {
      absl::Status status =
          CudaErrorStatus(error, flavor_ == AllocationFlavor::kDevice ? "free" : "free-host",
                          cuda_device_, health_);
      static_cast<void>(charge_.MarkLeaked());
      if (health_ != nullptr) health_->Poison(status);
      released_ = true;
      return status;
    }
    released_ = true;
    absl::Status status = charge_.active() ? charge_.Release() : absl::OkStatus();
    if (!status.ok() && health_ != nullptr) health_->Poison(status);
    return status;
  }

  void Abandon(void*, ByteCount, ByteCount, AllocationId id) noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_ || id != id_) return;
    if (charge_.active()) static_cast<void>(charge_.MarkLeaked());
    if (health_ != nullptr) {
      health_->Poison(WithErrorReason(
          absl::InternalError(
              "cuda.allocation: live allocation abandoned without explicit release"),
          ErrorReason::kInvariantViolation));
    }
    released_ = true;
  }

 private:
  const CudaApi* api_;
  CudaHealth* health_;
  DeviceId cuda_device_;
  AllocationId id_;
  AllocationFlavor flavor_;
  AllocationCharge charge_;
  std::mutex mutex_;
  bool released_ = false;
};

absl::StatusOr<Buffer> AllocateCuda(const AllocationRequest& request, DeviceId cuda_device,
                                    AllocationFlavor flavor, MemoryTracker& tracker,
                                    CudaHealth& health, const CudaApi& api) {
  absl::Status accepting = health.CheckAcceptingWork();
  if (!accepting.ok()) return accepting;
  absl::Status validation = ValidateAllocationRequest(request);
  if (!validation.ok()) return validation;
  if (flavor == AllocationFlavor::kDevice) {
    if (request.device != Device::Cuda(cuda_device) || request.memory_kind != MemoryKind::kDevice) {
      return absl::InvalidArgumentError(
          "cuda.allocator: device allocation request has wrong device or memory kind");
    }
  } else if (request.device != Device::Host() || request.memory_kind != MemoryKind::kPinnedHost) {
    return absl::InvalidArgumentError(
        "cuda.pinned_allocator: request must use host device and pinned memory");
  }
  absl::StatusOr<AllocationId> next_id = NextAllocationId();
  if (!next_id.ok()) return next_id.status();
  const AllocationId id = *next_id;
  std::shared_ptr<CudaAllocationDomain> domain;
  try {
    domain = std::make_shared<CudaAllocationDomain>(&api, &health, cuda_device, id, flavor);
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("cuda.allocation: allocation-domain construction failed");
  }
  if (request.bytes.value() != 0) {
    const MemoryKey key{request.device, request.memory_kind, request.category};
    absl::StatusOr<MemoryReservation> reservation = tracker.BeginReservation(key, request.bytes);
    if (!reservation.ok()) return reservation.status();
    absl::StatusOr<size_t> bytes =
        CheckedNarrow<size_t>(request.bytes.value(), "cuda.allocation.bytes");
    if (!bytes.ok()) return bytes.status();
    void* address = nullptr;
    const cudaError_t error = flavor == AllocationFlavor::kDevice
                                  ? api.malloc_device(&address, *bytes)
                                  : api.host_alloc(&address, *bytes, cudaHostAllocPortable);
    if (error != cudaSuccess) {
      return CudaErrorStatus(error, flavor == AllocationFlavor::kDevice ? "malloc" : "host-alloc",
                             cuda_device, &health);
    }
    ByteCount actual_alignment = PointerAlignment(address);
    if (actual_alignment.value() < request.alignment.value()) {
      return CleanupUncommittedAllocation(
          address, flavor, cuda_device, *reservation, health, api,
          absl::InvalidArgumentError(
              "cuda.allocation.alignment: runtime pointer does not satisfy request"));
    }
    absl::StatusOr<AllocationCharge> committed = reservation->Commit();
    if (!committed.ok()) {
      const cudaError_t cleanup_error = FreeRawAllocation(address, flavor, api);
      if (cleanup_error == cudaSuccess) return committed.status();
      absl::Status cleanup =
          CudaErrorStatus(cleanup_error, flavor == AllocationFlavor::kDevice ? "free" : "free-host",
                          cuda_device, &health);
      health.Poison(cleanup);
      return AnnotateCleanup(committed.status(), cleanup);
    }
    domain->AdoptCharge(std::move(*committed));
    absl::StatusOr<Buffer> buffer = Buffer::Adopt(address, request, actual_alignment, id, domain);
    if (!buffer.ok()) {
      return AnnotateCleanup(buffer.status(),
                             domain->Release(address, request.bytes, actual_alignment, id));
    }
    return buffer;
  }

  return Buffer::Adopt(nullptr, request, request.alignment, id, std::move(domain));
}

}  // namespace

CudaDeviceAllocator::CudaDeviceAllocator(DeviceId device, MemoryTracker& tracker,
                                         CudaHealth& health, const CudaApi& api) noexcept
    : device_(device), tracker_(&tracker), health_(&health), api_(&api) {}

CudaDeviceAllocator::CudaDeviceAllocator(CudaDeviceAllocator&& other) noexcept
    : device_(other.device_),
      tracker_(other.tracker_),
      health_(other.health_),
      api_(other.api_),
      active_(std::exchange(other.active_, false)) {
  other.tracker_ = nullptr;
  other.health_ = nullptr;
  other.api_ = nullptr;
}

CudaDeviceAllocator& CudaDeviceAllocator::operator=(CudaDeviceAllocator&& other) noexcept {
  if (this != &other) {
    device_ = other.device_;
    tracker_ = other.tracker_;
    health_ = other.health_;
    api_ = other.api_;
    active_ = std::exchange(other.active_, false);
    other.tracker_ = nullptr;
    other.health_ = nullptr;
    other.api_ = nullptr;
  }
  return *this;
}

absl::StatusOr<Buffer> CudaDeviceAllocator::Allocate(const AllocationRequest& request) {
  if (!active_) {
    return absl::FailedPreconditionError("cuda.allocator: allocator was moved");
  }
  return AllocateCuda(request, device_, AllocationFlavor::kDevice, *tracker_, *health_, *api_);
}

CudaPinnedAllocator::CudaPinnedAllocator(DeviceId owning_device, MemoryTracker& tracker,
                                         CudaHealth& health, const CudaApi& api) noexcept
    : owning_device_(owning_device), tracker_(&tracker), health_(&health), api_(&api) {}

CudaPinnedAllocator::CudaPinnedAllocator(CudaPinnedAllocator&& other) noexcept
    : owning_device_(other.owning_device_),
      tracker_(other.tracker_),
      health_(other.health_),
      api_(other.api_),
      active_(std::exchange(other.active_, false)) {
  other.tracker_ = nullptr;
  other.health_ = nullptr;
  other.api_ = nullptr;
}

CudaPinnedAllocator& CudaPinnedAllocator::operator=(CudaPinnedAllocator&& other) noexcept {
  if (this != &other) {
    owning_device_ = other.owning_device_;
    tracker_ = other.tracker_;
    health_ = other.health_;
    api_ = other.api_;
    active_ = std::exchange(other.active_, false);
    other.tracker_ = nullptr;
    other.health_ = nullptr;
    other.api_ = nullptr;
  }
  return *this;
}

absl::StatusOr<Buffer> CudaPinnedAllocator::Allocate(const AllocationRequest& request) {
  if (!active_) {
    return absl::FailedPreconditionError("cuda.pinned_allocator: allocator was moved");
  }
  return AllocateCuda(request, owning_device_, AllocationFlavor::kPinned, *tracker_, *health_,
                      *api_);
}

}  // namespace inferx::cuda
