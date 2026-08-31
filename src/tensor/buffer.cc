#include "inferx/tensor/buffer.h"

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"

namespace inferx {
namespace {

std::atomic<uint64_t> g_next_allocation_id{1};

bool IsPowerOfTwo(uint64_t value) { return value != 0 && std::has_single_bit(value); }

ByteCount SubviewAlignment(ByteCount alignment, uint64_t offset) {
  if (offset == 0) {
    return alignment;
  }
  const uint64_t offset_alignment = uint64_t{1} << std::countr_zero(offset);
  return ByteCount(alignment.value() < offset_alignment ? alignment.value() : offset_alignment);
}

const std::byte* AddOffset(const std::byte* address, uint64_t offset) {
  return address == nullptr ? nullptr : address + offset;
}

std::byte* AddOffset(std::byte* address, uint64_t offset) {
  return address == nullptr ? nullptr : address + offset;
}

}  // namespace

absl::StatusOr<AllocationId> NextAllocationId() {
  uint64_t value = g_next_allocation_id.load(std::memory_order_relaxed);
  while (value != std::numeric_limits<uint64_t>::max()) {
    if (g_next_allocation_id.compare_exchange_weak(value, value + 1, std::memory_order_relaxed,
                                                   std::memory_order_relaxed)) {
      return AllocationId(value);
    }
  }
  return absl::OutOfRangeError("buffer.allocation_id: identity space exhausted");
}

absl::Status ValidateAllocationRequest(const AllocationRequest& request) {
  absl::Status memory_status = ValidateMemoryKind(request.device, request.memory_kind);
  if (!memory_status.ok()) {
    return memory_status;
  }
  if (!IsPowerOfTwo(request.alignment.value())) {
    return absl::InvalidArgumentError("allocation.alignment: must be a nonzero power of two");
  }
  absl::StatusOr<size_t> bytes = CheckedNarrow<size_t>(request.bytes.value(), "allocation.bytes");
  if (!bytes.ok()) {
    return bytes.status();
  }
  absl::StatusOr<size_t> alignment =
      CheckedNarrow<size_t>(request.alignment.value(), "allocation.alignment");
  if (!alignment.ok()) {
    return alignment.status();
  }
  switch (request.category) {
    case MemoryCategory::kModelWeights:
    case MemoryCategory::kKvCache:
    case MemoryCategory::kWorkspace:
    case MemoryCategory::kExecutionMetadata:
    case MemoryCategory::kPinnedStaging:
    case MemoryCategory::kRuntimeInternal:
    case MemoryCategory::kTest:
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError("allocation.category: invalid memory category");
}

absl::Status ByteRange::ValidateWithin(ByteCount allocation_size) const {
  absl::StatusOr<uint64_t> end = CheckedAdd(offset.value(), size.value(), "buffer.range");
  if (!end.ok()) {
    return end.status();
  }
  if (*end > allocation_size.value()) {
    return absl::OutOfRangeError("buffer.range: range exceeds allocation");
  }
  return absl::OkStatus();
}

Buffer::Buffer() noexcept = default;

Buffer::Buffer(void* address, const AllocationRequest& request, ByteCount guaranteed_alignment,
               AllocationId allocation_id, std::shared_ptr<AllocationDomain> domain) noexcept
    : address_(address),
      size_(request.bytes),
      alignment_(guaranteed_alignment),
      device_(request.device),
      memory_kind_(request.memory_kind),
      category_(request.category),
      allocation_id_(allocation_id),
      domain_(std::move(domain)) {}

Buffer::~Buffer() noexcept {
  if (domain_ != nullptr) {
    if (size_.value() != 0) {
      domain_->Abandon(address_, size_, alignment_, allocation_id_);
    }
    Reset();
  }
}

Buffer::Buffer(Buffer&& other) noexcept
    : address_(std::exchange(other.address_, nullptr)),
      size_(std::exchange(other.size_, ByteCount(0))),
      alignment_(std::exchange(other.alignment_, ByteCount(1))),
      device_(std::exchange(other.device_, Device::Host())),
      memory_kind_(std::exchange(other.memory_kind_, MemoryKind::kHost)),
      category_(std::exchange(other.category_, MemoryCategory::kRuntimeInternal)),
      allocation_id_(std::exchange(other.allocation_id_, AllocationId(0))),
      domain_(std::move(other.domain_)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (domain_ != nullptr && size_.value() != 0) {
    domain_->Abandon(address_, size_, alignment_, allocation_id_);
  }
  address_ = std::exchange(other.address_, nullptr);
  size_ = std::exchange(other.size_, ByteCount(0));
  alignment_ = std::exchange(other.alignment_, ByteCount(1));
  device_ = std::exchange(other.device_, Device::Host());
  memory_kind_ = std::exchange(other.memory_kind_, MemoryKind::kHost);
  category_ = std::exchange(other.category_, MemoryCategory::kRuntimeInternal);
  allocation_id_ = std::exchange(other.allocation_id_, AllocationId(0));
  domain_ = std::move(other.domain_);
  return *this;
}

absl::StatusOr<Buffer> Buffer::Adopt(void* address, const AllocationRequest& request,
                                     ByteCount guaranteed_alignment, AllocationId allocation_id,
                                     std::shared_ptr<AllocationDomain> domain) {
  absl::Status status = ValidateAllocationRequest(request);
  if (!status.ok()) {
    return status;
  }
  if (domain == nullptr) {
    return absl::InvalidArgumentError("buffer.domain: allocation domain is required");
  }
  if (!IsPowerOfTwo(guaranteed_alignment.value()) ||
      guaranteed_alignment.value() < request.alignment.value()) {
    return absl::InvalidArgumentError(
        "buffer.alignment: guarantee must be a sufficient power of two");
  }
  if (request.bytes.value() != 0 && address == nullptr) {
    return absl::InvalidArgumentError("buffer.address: non-empty allocation has null address");
  }
  if (address != nullptr &&
      reinterpret_cast<uintptr_t>(address) % guaranteed_alignment.value() != 0) {
    return absl::InvalidArgumentError("buffer.address: address violates alignment guarantee");
  }
  return Buffer(address, request, guaranteed_alignment, allocation_id, std::move(domain));
}

absl::StatusOr<BufferView> Buffer::View(ByteRange range) const {
  if (domain_ == nullptr) {
    return absl::FailedPreconditionError("buffer.view: buffer has been released");
  }
  absl::Status status = range.ValidateWithin(size_);
  if (!status.ok()) {
    return status;
  }
  const auto* base = static_cast<const std::byte*>(address_);
  return BufferView(AddOffset(base, range.offset.value()), range,
                    SubviewAlignment(alignment_, range.offset.value()), device_, memory_kind_,
                    category_, allocation_id_);
}

absl::StatusOr<MutableBufferView> Buffer::MutableView(ByteRange range) {
  if (domain_ == nullptr) {
    return absl::FailedPreconditionError("buffer.view: buffer has been released");
  }
  absl::Status status = range.ValidateWithin(size_);
  if (!status.ok()) {
    return status;
  }
  auto* base = static_cast<std::byte*>(address_);
  return MutableBufferView(AddOffset(base, range.offset.value()), range,
                           SubviewAlignment(alignment_, range.offset.value()), device_,
                           memory_kind_, category_, allocation_id_);
}

absl::Status Buffer::Release() {
  if (domain_ == nullptr) {
    return absl::FailedPreconditionError("buffer.release: buffer already released");
  }
  if (size_.value() == 0) {
    Reset();
    return absl::OkStatus();
  }
  absl::Status status = domain_->Release(address_, size_, alignment_, allocation_id_);
  if (status.ok()) {
    Reset();
  }
  return status;
}

void Buffer::Reset() noexcept {
  address_ = nullptr;
  size_ = ByteCount(0);
  alignment_ = ByteCount(1);
  device_ = Device::Host();
  memory_kind_ = MemoryKind::kHost;
  category_ = MemoryCategory::kRuntimeInternal;
  allocation_id_ = AllocationId(0);
  domain_.reset();
}

BufferView::BufferView(const std::byte* address, ByteRange range, ByteCount alignment,
                       Device device, MemoryKind memory_kind, MemoryCategory category,
                       AllocationId allocation_id) noexcept
    : address_(address),
      range_(range),
      alignment_(alignment),
      device_(device),
      memory_kind_(memory_kind),
      category_(category),
      allocation_id_(allocation_id) {}

absl::StatusOr<std::span<const std::byte>> BufferView::HostBytes() const {
  if (!IsHostAddressable(memory_kind_)) {
    return absl::FailedPreconditionError("buffer.host_bytes: memory is not host addressable");
  }
  absl::StatusOr<size_t> size = CheckedNarrow<size_t>(range_.size.value(), "buffer.host_bytes");
  if (!size.ok()) {
    return size.status();
  }
  return std::span<const std::byte>(address_, *size);
}

absl::StatusOr<BufferView> BufferView::Subview(ByteRange range) const {
  absl::Status status = range.ValidateWithin(range_.size);
  if (!status.ok()) {
    return status;
  }
  absl::StatusOr<uint64_t> absolute =
      CheckedAdd(range_.offset.value(), range.offset.value(), "buffer.subview");
  if (!absolute.ok()) {
    return absolute.status();
  }
  return BufferView(AddOffset(address_, range.offset.value()),
                    ByteRange{ByteCount(*absolute), range.size},
                    SubviewAlignment(alignment_, range.offset.value()), device_, memory_kind_,
                    category_, allocation_id_);
}

MutableBufferView::MutableBufferView(std::byte* address, ByteRange range, ByteCount alignment,
                                     Device device, MemoryKind memory_kind, MemoryCategory category,
                                     AllocationId allocation_id) noexcept
    : address_(address),
      range_(range),
      alignment_(alignment),
      device_(device),
      memory_kind_(memory_kind),
      category_(category),
      allocation_id_(allocation_id) {}

absl::StatusOr<std::span<std::byte>> MutableBufferView::HostBytes() const {
  if (!IsHostAddressable(memory_kind_)) {
    return absl::FailedPreconditionError("buffer.host_bytes: memory is not host addressable");
  }
  absl::StatusOr<size_t> size = CheckedNarrow<size_t>(range_.size.value(), "buffer.host_bytes");
  if (!size.ok()) {
    return size.status();
  }
  return std::span<std::byte>(address_, *size);
}

absl::StatusOr<MutableBufferView> MutableBufferView::Subview(ByteRange range) const {
  absl::Status status = range.ValidateWithin(range_.size);
  if (!status.ok()) {
    return status;
  }
  absl::StatusOr<uint64_t> absolute =
      CheckedAdd(range_.offset.value(), range.offset.value(), "buffer.subview");
  if (!absolute.ok()) {
    return absolute.status();
  }
  return MutableBufferView(AddOffset(address_, range.offset.value()),
                           ByteRange{ByteCount(*absolute), range.size},
                           SubviewAlignment(alignment_, range.offset.value()), device_,
                           memory_kind_, category_, allocation_id_);
}

BufferView MutableBufferView::AsConst() const noexcept {
  return BufferView(address_, range_, alignment_, device_, memory_kind_, category_, allocation_id_);
}

}  // namespace inferx
