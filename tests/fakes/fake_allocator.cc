#include "tests/fakes/fake_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"
#include "inferx/tensor/buffer.h"

namespace inferx::testing {

struct FakeAllocator::State {
  std::mutex mutex;
  ByteCount capacity = ByteCount(0);
  ByteCount live = ByteCount(0);
  ByteCount reserved = ByteCount(0);
  uint64_t allocations = 0;
  std::optional<FakeAllocatorFailure> failure;
};

namespace {

class FakeAllocationDomain final : public AllocationDomain {
 public:
  explicit FakeAllocationDomain(std::shared_ptr<FakeAllocator::State> state)
      : state_(std::move(state)) {}

  absl::Status Release(void* address, ByteCount size, ByteCount alignment, AllocationId) override {
    ::operator delete(address, std::align_val_t(alignment.value()));
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->live = ByteCount(state_->live.value() - size.value());
    return absl::OkStatus();
  }

  void Abandon(void* address, ByteCount size, ByteCount alignment,
               AllocationId id) noexcept override {
    static_cast<void>(Release(address, size, alignment, id));
  }

 private:
  std::shared_ptr<FakeAllocator::State> state_;
};

}  // namespace

FakeAllocator::FakeAllocator(ByteCount capacity) : state_(std::make_shared<State>()) {
  state_->capacity = capacity;
}
FakeAllocator::~FakeAllocator() = default;
FakeAllocator::FakeAllocator(FakeAllocator&&) noexcept = default;
FakeAllocator& FakeAllocator::operator=(FakeAllocator&&) noexcept = default;

void FakeAllocator::SetFailure(FakeAllocatorFailure failure) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->failure = failure;
}

void FakeAllocator::ClearFailure() noexcept {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->failure.reset();
}

uint64_t FakeAllocator::allocation_count() const noexcept {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->allocations;
}

ByteCount FakeAllocator::live_bytes() const noexcept {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->live;
}

absl::StatusOr<Buffer> FakeAllocator::Allocate(const AllocationRequest& request) {
  absl::Status validation = ValidateAllocationRequest(request);
  if (!validation.ok() && request.memory_kind != MemoryKind::kManaged) {
    return validation;
  }
  if (request.memory_kind == MemoryKind::kManaged) {
    return absl::UnimplementedError("fake_allocator: managed memory is unsupported");
  }
  std::shared_ptr<FakeAllocationDomain> domain;
  try {
    domain = std::make_shared<FakeAllocationDomain>(state_);
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("fake_allocator: allocation-domain construction failed");
  }
  uint64_t ordinal = 0;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ordinal = ++state_->allocations;
    const std::optional<FakeAllocatorFailure> failure_config = state_->failure;
    if (failure_config.has_value()) {
      const FakeAllocatorFailure& failure = failure_config.value();
      const bool ordinal_matches =
          !failure.allocation_ordinal.has_value() || *failure.allocation_ordinal == ordinal;
      const bool category_matches =
          !failure.category.has_value() || *failure.category == request.category;
      const bool threshold_matches = !failure.bytes_at_least.has_value() ||
                                     request.bytes.value() >= failure.bytes_at_least->value();
      if (ordinal_matches && category_matches && threshold_matches) {
        return absl::ResourceExhaustedError("fake_allocator: injected allocation failure");
      }
    }
    const uint64_t available =
        state_->capacity.value() - state_->live.value() - state_->reserved.value();
    if (request.bytes.value() > available) {
      return absl::ResourceExhaustedError("fake_allocator: capacity exhausted");
    }
    state_->reserved = ByteCount(state_->reserved.value() + request.bytes.value());
  }
  const uint64_t actual_alignment = request.alignment.value() < alignof(std::max_align_t)
                                        ? alignof(std::max_align_t)
                                        : request.alignment.value();
  absl::StatusOr<size_t> bytes =
      CheckedNarrow<size_t>(request.bytes.value(), "fake_allocator.bytes");
  if (!bytes.ok()) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->reserved = ByteCount(state_->reserved.value() - request.bytes.value());
    return bytes.status();
  }
  void* address = nullptr;
  if (*bytes != 0) {
    try {
      address = ::operator new(*bytes, std::align_val_t(actual_alignment));
    } catch (const std::bad_alloc&) {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->reserved = ByteCount(state_->reserved.value() - request.bytes.value());
      return absl::ResourceExhaustedError("fake_allocator: host storage failed");
    }
  }
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->reserved = ByteCount(state_->reserved.value() - request.bytes.value());
    state_->live = ByteCount(state_->live.value() + request.bytes.value());
  }
  absl::StatusOr<AllocationId> id = NextAllocationId();
  if (!id.ok()) {
    if (address != nullptr) {
      ::operator delete(address, std::align_val_t(actual_alignment));
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->live = ByteCount(state_->live.value() - request.bytes.value());
    return id.status();
  }
  absl::StatusOr<Buffer> buffer =
      Buffer::Adopt(address, request, ByteCount(actual_alignment), *id, domain);
  if (!buffer.ok()) {
    static_cast<void>(domain->Release(address, request.bytes, ByteCount(actual_alignment), *id));
    return buffer.status();
  }
  return buffer;
}

}  // namespace inferx::testing
