// Device-neutral allocation contract.
#ifndef INFERX_TENSOR_ALLOCATOR_H_
#define INFERX_TENSOR_ALLOCATOR_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/token.h"
#include "inferx/tensor/device.h"

namespace inferx {

class Buffer;

enum class MemoryCategory : uint8_t {
  kModelWeights,
  kKvCache,
  kWorkspace,
  kExecutionMetadata,
  kPinnedStaging,
  kRuntimeInternal,
  kTest,
};

struct AllocationRequest {
  Device device;
  MemoryKind memory_kind;
  ByteCount bytes;
  ByteCount alignment;
  MemoryCategory category;
};

[[nodiscard]] absl::Status ValidateAllocationRequest(const AllocationRequest& request);

// Device-neutral transaction hook used by allocators that optionally
// participate in runtime memory accounting. The interface lives in tensor so
// runtime can implement it without reversing the tensor <- runtime dependency.
class AllocationAccountingReservation {
 public:
  virtual ~AllocationAccountingReservation() = default;
  [[nodiscard]] virtual absl::Status Commit() = 0;
  [[nodiscard]] virtual absl::Status Release() = 0;
  virtual void Abandon() noexcept = 0;
};

class AllocationAccounting {
 public:
  virtual ~AllocationAccounting() = default;
  [[nodiscard]] virtual absl::StatusOr<std::unique_ptr<AllocationAccountingReservation>>
  BeginAllocationReservation(const AllocationRequest& request) = 0;
};

class Allocator {
 public:
  virtual ~Allocator() = default;
  [[nodiscard]] virtual absl::StatusOr<Buffer> Allocate(const AllocationRequest& request) = 0;
};

class CpuAllocator final : public Allocator {
 public:
  CpuAllocator();
  explicit CpuAllocator(AllocationAccounting& accounting);
  ~CpuAllocator() override;

  CpuAllocator(const CpuAllocator&) = delete;
  CpuAllocator& operator=(const CpuAllocator&) = delete;
  CpuAllocator(CpuAllocator&&) noexcept;
  CpuAllocator& operator=(CpuAllocator&&) noexcept;

  [[nodiscard]] absl::StatusOr<Buffer> Allocate(const AllocationRequest& request) override;

 private:
  class Impl;
  Impl* impl_;
};

}  // namespace inferx

#endif  // INFERX_TENSOR_ALLOCATOR_H_
