#ifndef INFERX_TESTS_FAKES_FAKE_ALLOCATOR_H_
#define INFERX_TESTS_FAKES_FAKE_ALLOCATOR_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "inferx/base/token.h"
#include "inferx/tensor/allocator.h"

namespace inferx::testing {

struct FakeAllocatorFailure {
  std::optional<uint64_t> allocation_ordinal;
  std::optional<MemoryCategory> category;
  std::optional<ByteCount> bytes_at_least;
};

class FakeAllocator final : public Allocator {
 public:
  struct State;
  explicit FakeAllocator(ByteCount capacity);
  ~FakeAllocator() override;
  FakeAllocator(FakeAllocator&&) noexcept;
  FakeAllocator& operator=(FakeAllocator&&) noexcept;
  FakeAllocator(const FakeAllocator&) = delete;
  FakeAllocator& operator=(const FakeAllocator&) = delete;

  void SetFailure(FakeAllocatorFailure failure);
  void ClearFailure() noexcept;
  [[nodiscard]] uint64_t allocation_count() const noexcept;
  [[nodiscard]] ByteCount live_bytes() const noexcept;
  [[nodiscard]] absl::StatusOr<Buffer> Allocate(const AllocationRequest& request) override;

 private:
  std::shared_ptr<State> state_;
};

}  // namespace inferx::testing

#endif  // INFERX_TESTS_FAKES_FAKE_ALLOCATOR_H_
