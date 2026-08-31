#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/runtime/buffer_pool.h"
#include "inferx/runtime/completion_fence.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/tensor_view.h"
#include "tests/fakes/fake_fence.h"

namespace inferx {
namespace {

TEST(M2RuntimeStressTest, OneHundredThousandGenerationCyclesReturnBaseline) {
  CpuAllocator allocator;
  const AllocationRequest request{Device::Host(), MemoryKind::kHost, ByteCount(64), ByteCount(64),
                                  MemoryCategory::kTest};
  Buffer backing = allocator.Allocate(request).value();
  FixedBufferPool pool =
      FixedBufferPool::Create(std::move(backing),
                              PoolGeometry{1, ByteCount(64), ByteCount(64), PoolGeneration(0)})
          .value();
  testing::FakeFenceDomain fences(1);
  for (uint32_t cycle = 0; cycle < 100000; ++cycle) {
    BufferLease lease = pool.Acquire().value();
    CompletionFence fence = fences.Acquire().value();
    EXPECT_TRUE(fences.Complete(fence.token()).ok());
    EXPECT_EQ(fence.Poll()->state, FenceState::kComplete);
    EXPECT_TRUE(fence.Acknowledge().ok());
    EXPECT_TRUE(lease.Release().ok());
  }
  EXPECT_EQ(pool.available_slots(), 1);
  EXPECT_TRUE(pool.ValidateInvariants().ok());
  EXPECT_TRUE(pool.Close().ok());
}

TEST(M2RuntimeStressTest, TenThousandCpuReferenceTransformationsAreExact) {
  constexpr uint64_t kStorageBytes = 1024;
  CpuAllocator allocator;
  const AllocationRequest request{Device::Host(), MemoryKind::kHost, ByteCount(kStorageBytes),
                                  ByteCount(64), MemoryCategory::kTest};
  Buffer source_buffer = allocator.Allocate(request).value();
  Buffer destination_buffer = allocator.Allocate(request).value();
  MutableBufferView source_storage =
      source_buffer.MutableView({ByteCount(0), ByteCount(kStorageBytes)}).value();
  MutableBufferView destination_storage =
      destination_buffer.MutableView({ByteCount(0), ByteCount(kStorageBytes)}).value();
  std::span<std::byte> source_bytes = source_storage.HostBytes().value();
  std::span<std::byte> destination_bytes = destination_storage.HostBytes().value();

  for (uint64_t fixture = 0; fixture < 10000; ++fixture) {
    const uint64_t rows = 1 + fixture % 8;
    const uint64_t columns = 1 + (fixture / 8) % 8;
    const uint64_t source_padding = fixture % 5;
    const uint64_t destination_padding = (fixture / 5) % 5;
    const std::array<uint64_t, 2> dimensions{rows, columns};
    const std::array<uint64_t, 2> source_strides{columns + source_padding, 1};
    const std::array<uint64_t, 2> destination_strides{1, rows + destination_padding};
    const Shape shape = Shape::Create(dimensions).value();
    const TensorView source = TensorView::Create(source_storage.AsConst(), DType::kUInt8, shape,
                                                 Strides::CreateElements(source_strides).value())
                                  .value();
    const MutableTensorView destination =
        MutableTensorView::Create(destination_storage, DType::kUInt8, shape,
                                  Strides::CreateElements(destination_strides).value())
            .value();

    ASSERT_TRUE(FillBufferCpu(source_storage, 0).ok());
    ASSERT_TRUE(FillBufferCpu(destination_storage, 0xff).ok());
    for (uint64_t row = 0; row < rows; ++row) {
      for (uint64_t column = 0; column < columns; ++column) {
        const uint8_t value = static_cast<uint8_t>((fixture + row * columns + column) % 251);
        source_bytes[row * source_strides[0] + column] = static_cast<std::byte>(value);
      }
    }
    ASSERT_TRUE(CopyTensorCpu(source, destination).ok());
    for (uint64_t row = 0; row < rows; ++row) {
      for (uint64_t column = 0; column < columns; ++column) {
        EXPECT_EQ(destination_bytes[row + column * destination_strides[1]],
                  source_bytes[row * source_strides[0] + column]);
      }
    }
  }
  EXPECT_TRUE(source_buffer.Release().ok());
  EXPECT_TRUE(destination_buffer.Release().ok());
}

TEST(M2RuntimeStressTest, MutexAdapterSupportsMpmcLeaseCycles) {
  constexpr uint32_t kSlotCount = 8;
  constexpr uint32_t kThreadCount = 4;
  constexpr uint32_t kCyclesPerThread = 5000;
  CpuAllocator allocator;
  Buffer backing = allocator
                       .Allocate({Device::Host(), MemoryKind::kHost,
                                  ByteCount(static_cast<uint64_t>(kSlotCount) * 64), ByteCount(64),
                                  MemoryCategory::kTest})
                       .value();
  MutexFixedBufferPool pool(
      FixedBufferPool::Create(std::move(backing), PoolGeometry{kSlotCount, ByteCount(64),
                                                               ByteCount(64), PoolGeneration(0)})
          .value());
  std::atomic<bool> failed = false;
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (uint32_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    workers.emplace_back([&pool, &failed, thread_index]() {
      for (uint32_t cycle = 0; cycle < kCyclesPerThread; ++cycle) {
        absl::StatusOr<BufferLease> lease = pool.Acquire();
        if (!lease.ok()) {
          failed.store(true, std::memory_order_relaxed);
          return;
        }
        absl::StatusOr<MutableBufferView> view = lease->mutable_view();
        if (!view.ok() || !FillBufferCpu(*view, static_cast<uint8_t>(thread_index)).ok() ||
            !lease->Release().ok()) {
          failed.store(true, std::memory_order_relaxed);
          return;
        }
      }
    });
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  EXPECT_FALSE(failed.load(std::memory_order_relaxed));
  EXPECT_TRUE(pool.ValidateInvariants().ok());
}

}  // namespace
}  // namespace inferx
