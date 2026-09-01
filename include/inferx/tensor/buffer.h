// Move-only allocation ownership and bounded non-owning byte views.
#ifndef INFERX_TENSOR_BUFFER_H_
#define INFERX_TENSOR_BUFFER_H_

#include <cstddef>
#include <memory>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/device.h"

namespace inferx::cuda {
class BufferAccess;
}

namespace inferx {

struct ByteRange {
  ByteCount offset;
  ByteCount size;

  [[nodiscard]] absl::Status ValidateWithin(ByteCount allocation_size) const;
};

class AllocationDomain {
 public:
  virtual ~AllocationDomain() = default;
  [[nodiscard]] virtual absl::Status Release(void* address, ByteCount size, ByteCount alignment,
                                             AllocationId allocation_id) = 0;
  virtual void Abandon(void* address, ByteCount size, ByteCount alignment,
                       AllocationId allocation_id) noexcept = 0;
};

// Process-unique, checked allocation identity. Allocators use this instead of
// pointer identity or allocator-local counters.
[[nodiscard]] absl::StatusOr<AllocationId> NextAllocationId();

class BufferView;
class MutableBufferView;

class Buffer {
 public:
  Buffer() noexcept;
  ~Buffer() noexcept;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  [[nodiscard]] static absl::StatusOr<Buffer> Adopt(void* address, const AllocationRequest& request,
                                                    ByteCount guaranteed_alignment,
                                                    AllocationId allocation_id,
                                                    std::shared_ptr<AllocationDomain> domain);

  [[nodiscard]] bool empty() const noexcept { return domain_ == nullptr; }
  [[nodiscard]] ByteCount size() const noexcept { return size_; }
  [[nodiscard]] ByteCount alignment() const noexcept { return alignment_; }
  [[nodiscard]] Device device() const noexcept { return device_; }
  [[nodiscard]] MemoryKind memory_kind() const noexcept { return memory_kind_; }
  [[nodiscard]] MemoryCategory category() const noexcept { return category_; }
  [[nodiscard]] AllocationId allocation_id() const noexcept { return allocation_id_; }

  [[nodiscard]] absl::StatusOr<BufferView> View(ByteRange range) const;
  [[nodiscard]] absl::StatusOr<MutableBufferView> MutableView(ByteRange range);
  [[nodiscard]] absl::Status Release();

 private:
  Buffer(void* address, const AllocationRequest& request, ByteCount guaranteed_alignment,
         AllocationId allocation_id, std::shared_ptr<AllocationDomain> domain) noexcept;
  void Reset() noexcept;

  void* address_ = nullptr;
  ByteCount size_ = ByteCount(0);
  ByteCount alignment_ = ByteCount(1);
  Device device_ = Device::Host();
  MemoryKind memory_kind_ = MemoryKind::kHost;
  MemoryCategory category_ = MemoryCategory::kRuntimeInternal;
  AllocationId allocation_id_ = AllocationId(0);
  std::shared_ptr<AllocationDomain> domain_;
};

class BufferView {
 public:
  [[nodiscard]] ByteCount size() const noexcept { return range_.size; }
  [[nodiscard]] ByteCount alignment() const noexcept { return alignment_; }
  [[nodiscard]] ByteRange range() const noexcept { return range_; }
  [[nodiscard]] Device device() const noexcept { return device_; }
  [[nodiscard]] MemoryKind memory_kind() const noexcept { return memory_kind_; }
  [[nodiscard]] MemoryCategory category() const noexcept { return category_; }
  [[nodiscard]] AllocationId allocation_id() const noexcept { return allocation_id_; }
  [[nodiscard]] absl::StatusOr<std::span<const std::byte>> HostBytes() const;
  [[nodiscard]] absl::StatusOr<BufferView> Subview(ByteRange range) const;

 private:
  friend class Buffer;
  friend class MutableBufferView;
  friend class cuda::BufferAccess;
  BufferView(const std::byte* address, ByteRange range, ByteCount alignment, Device device,
             MemoryKind memory_kind, MemoryCategory category, AllocationId allocation_id) noexcept;

  const std::byte* address_ = nullptr;
  ByteRange range_{ByteCount(0), ByteCount(0)};
  ByteCount alignment_ = ByteCount(1);
  Device device_ = Device::Host();
  MemoryKind memory_kind_ = MemoryKind::kHost;
  MemoryCategory category_ = MemoryCategory::kRuntimeInternal;
  AllocationId allocation_id_ = AllocationId(0);
};

class MutableBufferView {
 public:
  [[nodiscard]] ByteCount size() const noexcept { return range_.size; }
  [[nodiscard]] ByteCount alignment() const noexcept { return alignment_; }
  [[nodiscard]] ByteRange range() const noexcept { return range_; }
  [[nodiscard]] Device device() const noexcept { return device_; }
  [[nodiscard]] MemoryKind memory_kind() const noexcept { return memory_kind_; }
  [[nodiscard]] MemoryCategory category() const noexcept { return category_; }
  [[nodiscard]] AllocationId allocation_id() const noexcept { return allocation_id_; }
  [[nodiscard]] absl::StatusOr<std::span<std::byte>> HostBytes() const;
  [[nodiscard]] absl::StatusOr<MutableBufferView> Subview(ByteRange range) const;
  [[nodiscard]] BufferView AsConst() const noexcept;

 private:
  friend class Buffer;
  friend class cuda::BufferAccess;
  MutableBufferView(std::byte* address, ByteRange range, ByteCount alignment, Device device,
                    MemoryKind memory_kind, MemoryCategory category,
                    AllocationId allocation_id) noexcept;

  std::byte* address_ = nullptr;
  ByteRange range_{ByteCount(0), ByteCount(0)};
  ByteCount alignment_ = ByteCount(1);
  Device device_ = Device::Host();
  MemoryKind memory_kind_ = MemoryKind::kHost;
  MemoryCategory category_ = MemoryCategory::kRuntimeInternal;
  AllocationId allocation_id_ = AllocationId(0);
};

}  // namespace inferx

#endif  // INFERX_TENSOR_BUFFER_H_
