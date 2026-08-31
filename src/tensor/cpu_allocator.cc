#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <utility>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"

namespace inferx {
namespace {

class CpuAllocationDomain final : public AllocationDomain {
 public:
  CpuAllocationDomain(AllocationId id, std::unique_ptr<AllocationAccountingReservation> accounting)
      : id_(id), accounting_(std::move(accounting)) {}

  absl::Status CommitAccounting() {
    return accounting_ == nullptr ? absl::OkStatus() : accounting_->Commit();
  }

  absl::Status Release(void* address, ByteCount, ByteCount alignment, AllocationId id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_ || id != id_) {
      return absl::FailedPreconditionError(
          "cpu_allocator.release: duplicate or foreign allocation ID");
    }
    ::operator delete(address, std::align_val_t(alignment.value()));
    absl::Status status = accounting_ == nullptr ? absl::OkStatus() : accounting_->Release();
    if (status.ok()) released_ = true;
    return status;
  }

  void Abandon(void* address, ByteCount, ByteCount alignment, AllocationId id) noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_ || id != id_) return;
    ::operator delete(address, std::align_val_t(alignment.value()));
    if (accounting_ != nullptr) accounting_->Abandon();
    released_ = true;
  }

 private:
  AllocationId id_;
  std::unique_ptr<AllocationAccountingReservation> accounting_;
  std::mutex mutex_;
  bool released_ = false;
};

}  // namespace

class CpuAllocator::Impl {
 public:
  explicit Impl(AllocationAccounting* supplied_accounting) : accounting(supplied_accounting) {}
  AllocationAccounting* accounting = nullptr;
};

CpuAllocator::CpuAllocator() : impl_(new Impl(nullptr)) {}
CpuAllocator::CpuAllocator(AllocationAccounting& accounting) : impl_(new Impl(&accounting)) {}
CpuAllocator::~CpuAllocator() { delete impl_; }

CpuAllocator::CpuAllocator(CpuAllocator&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

CpuAllocator& CpuAllocator::operator=(CpuAllocator&& other) noexcept {
  if (this != &other) {
    delete impl_;
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

absl::StatusOr<Buffer> CpuAllocator::Allocate(const AllocationRequest& request) {
  if (impl_ == nullptr) {
    return absl::FailedPreconditionError("cpu_allocator: allocator was moved");
  }
  absl::Status status = ValidateAllocationRequest(request);
  if (!status.ok()) {
    return status;
  }
  if (request.device != Device::Host() || request.memory_kind != MemoryKind::kHost) {
    return absl::InvalidArgumentError(
        "cpu_allocator.memory_kind: only ordinary host memory is supported");
  }
  const uint64_t requested_alignment = request.alignment.value();
  const uint64_t actual_alignment = requested_alignment < alignof(std::max_align_t)
                                        ? alignof(std::max_align_t)
                                        : requested_alignment;
  absl::StatusOr<size_t> bytes =
      CheckedNarrow<size_t>(request.bytes.value(), "cpu_allocator.bytes");
  if (!bytes.ok()) {
    return bytes.status();
  }
  absl::StatusOr<size_t> alignment =
      CheckedNarrow<size_t>(actual_alignment, "cpu_allocator.alignment");
  if (!alignment.ok()) {
    return alignment.status();
  }
  absl::StatusOr<AllocationId> id = NextAllocationId();
  if (!id.ok()) return id.status();
  std::unique_ptr<AllocationAccountingReservation> accounting;
  if (impl_->accounting != nullptr && request.bytes.value() != 0) {
    absl::StatusOr<std::unique_ptr<AllocationAccountingReservation>> reserved =
        impl_->accounting->BeginAllocationReservation(request);
    if (!reserved.ok()) return reserved.status();
    accounting = std::move(*reserved);
  }
  std::shared_ptr<CpuAllocationDomain> domain;
  try {
    domain = std::make_shared<CpuAllocationDomain>(*id, std::move(accounting));
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("cpu_allocator: allocation-domain construction failed");
  }
  void* address = nullptr;
  if (*bytes != 0) {
    try {
      address = ::operator new(*bytes, std::align_val_t(*alignment));
    } catch (const std::bad_alloc&) {
      return absl::ResourceExhaustedError("cpu_allocator: aligned allocation failed");
    }
  }
  absl::Status committed = domain->CommitAccounting();
  if (!committed.ok()) {
    ::operator delete(address, std::align_val_t(*alignment));
    return committed;
  }
  absl::StatusOr<Buffer> buffer =
      Buffer::Adopt(address, request, ByteCount(actual_alignment), *id, domain);
  if (!buffer.ok()) {
    static_cast<void>(domain->Release(address, request.bytes, ByteCount(*alignment), *id));
  }
  return buffer;
}

}  // namespace inferx
