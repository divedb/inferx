#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/base/clock.h"
#include "inferx/base/status.h"
#include "inferx/runtime/buffer_pool.h"
#include "inferx/runtime/completion_fence.h"
#include "inferx/runtime/memory_tracker.h"
#include "inferx/runtime/workspace.h"
#include "inferx/tensor/allocator.h"
#include "tests/fakes/fake_allocator.h"
#include "tests/fakes/fake_fence.h"

namespace inferx {
namespace {

AllocationRequest HostRequest(uint64_t bytes, MemoryCategory category, uint64_t alignment = 64) {
  return AllocationRequest{Device::Host(), MemoryKind::kHost, ByteCount(bytes),
                           ByteCount(alignment), category};
}

TEST(M2ErrorReasonTest, CudaAndLifetimeReasonNamesRoundTrip) {
  constexpr std::array<ErrorReason, 11> kReasons{
      ErrorReason::kCudaInvalidDevice,
      ErrorReason::kCudaOutOfMemory,
      ErrorReason::kCudaLaunchRejected,
      ErrorReason::kCudaAsyncFault,
      ErrorReason::kCudaDeviceLost,
      ErrorReason::kCudaApiFailure,
      ErrorReason::kStaleFence,
      ErrorReason::kStalePoolLease,
      ErrorReason::kPoolExhausted,
      ErrorReason::kPendingResource,
      ErrorReason::kUnsupportedCapability,
  };
  for (ErrorReason reason : kReasons) {
    EXPECT_EQ(ErrorReasonFromName(ErrorReasonToName(reason)), reason);
  }
  EXPECT_FALSE(ErrorReasonFromName("unknown").has_value());
}

TEST(M2MemoryTrackerTest, ReservationCommitRollbackPeakAndLimit) {
  constexpr std::array<MemoryLimit, 1> kLimits{
      MemoryLimit{Device::Host(), MemoryKind::kHost, ByteCount(100)}};
  MemoryTracker tracker = MemoryTracker::Create(kLimits).value();
  const MemoryKey key{Device::Host(), MemoryKind::kHost, MemoryCategory::kTest};
  MemoryReservation reservation = tracker.BeginReservation(key, ByteCount(60)).value();
  EXPECT_EQ(tracker.Snapshot()[0].second.reserved.value(), 60);
  EXPECT_EQ(tracker.BeginReservation(key, ByteCount(41)).status().code(),
            absl::StatusCode::kResourceExhausted);
  AllocationCharge charge = reservation.Commit().value();
  EXPECT_EQ(tracker.Snapshot()[0].second.committed.value(), 60);
  EXPECT_EQ(tracker.Snapshot()[0].second.peak_committed.value(), 60);
  EXPECT_TRUE(charge.Release().ok());

  MemoryReservation rollback = tracker.BeginReservation(key, ByteCount(25)).value();
  EXPECT_TRUE(rollback.Rollback().ok());
  EXPECT_TRUE(tracker.ValidateBaseline().ok());
}

TEST(M2MemoryTrackerTest, LeakedChargeRemainsDiagnostic) {
  constexpr std::array<MemoryLimit, 1> kLimits{
      MemoryLimit{Device::Host(), MemoryKind::kHost, ByteCount(100)}};
  MemoryTracker tracker = MemoryTracker::Create(kLimits).value();
  const MemoryKey key{Device::Host(), MemoryKind::kHost, MemoryCategory::kTest};
  AllocationCharge charge = tracker.BeginReservation(key, ByteCount(12))->Commit().value();
  EXPECT_TRUE(charge.MarkLeaked().ok());
  EXPECT_EQ(tracker.Snapshot()[0].second.leaked.value(), 12);
  EXPECT_FALSE(tracker.ValidateBaseline().ok());
  EXPECT_TRUE(tracker.ValidateBaseline(true).ok());
}

TEST(M2MemoryTrackerTest, ConcurrentSnapshotsRemainWithinDomainLimit) {
  constexpr std::array<MemoryLimit, 1> kLimits{
      MemoryLimit{Device::Host(), MemoryKind::kHost, ByteCount(64)}};
  MemoryTracker tracker = MemoryTracker::Create(kLimits).value();
  const MemoryKey key{Device::Host(), MemoryKind::kHost, MemoryCategory::kTest};
  std::atomic<uint32_t> running = 4;
  std::atomic<bool> failed = false;
  std::vector<std::thread> workers;
  workers.reserve(4);
  for (uint32_t worker = 0; worker < 4; ++worker) {
    static_cast<void>(worker);
    workers.emplace_back([&tracker, &key, &running, &failed]() {
      for (uint32_t iteration = 0; iteration < 1000; ++iteration) {
        absl::StatusOr<MemoryReservation> reservation = tracker.BeginReservation(key, ByteCount(1));
        if (!reservation.ok()) {
          failed.store(true, std::memory_order_relaxed);
          break;
        }
        absl::StatusOr<AllocationCharge> charge = reservation->Commit();
        if (!charge.ok() || !charge->Release().ok()) {
          failed.store(true, std::memory_order_relaxed);
          break;
        }
      }
      running.fetch_sub(1, std::memory_order_release);
    });
  }
  while (running.load(std::memory_order_acquire) != 0) {
    for (const auto& [snapshot_key, counters] : tracker.Snapshot()) {
      static_cast<void>(snapshot_key);
      EXPECT_LE(counters.reserved.value() + counters.committed.value() + counters.leaked.value(),
                counters.limit.value());
      EXPECT_LE(counters.live_allocations, 4);
    }
  }
  for (std::thread& worker : workers) worker.join();
  EXPECT_FALSE(failed.load(std::memory_order_relaxed));
  EXPECT_TRUE(tracker.ValidateBaseline().ok());
}

TEST(M2MemoryTrackerTest, CpuAllocatorUsesSuppliedTransactionalAccounting) {
  const MemoryLimit limits[] = {{Device::Host(), MemoryKind::kHost, ByteCount(128)}};
  MemoryTracker tracker = MemoryTracker::Create(limits).value();
  CpuAllocator allocator(tracker);

  Buffer buffer = allocator
                      .Allocate({Device::Host(), MemoryKind::kHost, ByteCount(96), ByteCount(64),
                                 MemoryCategory::kTest})
                      .value();
  const auto active = tracker.Snapshot();
  ASSERT_EQ(active.size(), 1);
  EXPECT_EQ(active.front().second.reserved.value(), 0);
  EXPECT_EQ(active.front().second.committed.value(), 96);
  EXPECT_EQ(active.front().second.live_allocations, 1);

  const auto exhausted = allocator.Allocate({Device::Host(), MemoryKind::kHost, ByteCount(64),
                                             ByteCount(64), MemoryCategory::kWorkspace});
  EXPECT_EQ(exhausted.status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_TRUE(buffer.Release().ok());
  EXPECT_TRUE(tracker.ValidateBaseline().ok());
}

TEST(M2FakeAllocatorTest, DeviceMetadataAndFailureHooksAreDeterministic) {
  testing::FakeAllocator allocator(ByteCount(128));
  testing::FakeAllocatorFailure failure;
  failure.allocation_ordinal = 2;
  allocator.SetFailure(failure);
  Buffer first = allocator.Allocate(HostRequest(32, MemoryCategory::kTest)).value();
  const AllocationRequest fake_device{Device::Cuda(DeviceId(3)), MemoryKind::kDevice, ByteCount(32),
                                      ByteCount(16), MemoryCategory::kTest};
  EXPECT_EQ(allocator.Allocate(fake_device).status().code(), absl::StatusCode::kResourceExhausted);
  allocator.ClearFailure();
  Buffer device = allocator.Allocate(fake_device).value();
  EXPECT_EQ(device.View({ByteCount(0), ByteCount(32)})->HostBytes().status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(allocator.live_bytes().value(), 64);
  EXPECT_TRUE(device.Release().ok());
  EXPECT_TRUE(first.Release().ok());
  EXPECT_EQ(allocator.live_bytes().value(), 0);
}

TEST(M2BufferPoolTest, LowestSlotExhaustionGenerationAndClose) {
  CpuAllocator allocator;
  Buffer backing = allocator.Allocate(HostRequest(128, MemoryCategory::kRuntimeInternal)).value();
  FixedBufferPool pool =
      FixedBufferPool::Create(std::move(backing),
                              PoolGeometry{2, ByteCount(32), ByteCount(32), PoolGeneration(0)})
          .value();
  BufferLease first = pool.Acquire().value();
  BufferLease second = pool.Acquire().value();
  EXPECT_EQ(first.token().slot.value(), 0);
  EXPECT_EQ(second.token().slot.value(), 1);
  absl::Status full = pool.Acquire().status();
  EXPECT_EQ(full.code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(GetErrorReason(full).value(), ErrorReason::kPoolExhausted);
  const PoolGeneration first_generation = first.token().generation;
  EXPECT_TRUE(first.Release().ok());
  EXPECT_EQ(first.Release().code(), absl::StatusCode::kFailedPrecondition);
  BufferLease reused = pool.Acquire().value();
  EXPECT_EQ(reused.token().slot.value(), 0);
  EXPECT_GT(reused.token().generation.value(), first_generation.value());
  EXPECT_FALSE(pool.Close().ok());
  EXPECT_TRUE(second.Release().ok());
  EXPECT_TRUE(reused.Release().ok());
  EXPECT_TRUE(pool.ValidateInvariants().ok());
  EXPECT_TRUE(pool.Close().ok());
}

TEST(M2BufferPoolTest, MaximumGenerationRetiresSlot) {
  CpuAllocator allocator;
  Buffer backing = allocator.Allocate(HostRequest(64, MemoryCategory::kRuntimeInternal)).value();
  FixedBufferPool pool =
      FixedBufferPool::Create(std::move(backing),
                              PoolGeometry{1, ByteCount(64), ByteCount(64),
                                           PoolGeneration(std::numeric_limits<uint64_t>::max())})
          .value();
  EXPECT_EQ(pool.Acquire().status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(pool.available_slots(), 0);
  EXPECT_TRUE(pool.ValidateInvariants().ok());
  EXPECT_TRUE(pool.Close().ok());
}

TEST(M2LifetimeDeathTest, TrackerDestructionRejectsLiveCharges) {
  EXPECT_DEATH(([] {
                 const std::array<MemoryLimit, 1> limits{
                     MemoryLimit{Device::Host(), MemoryKind::kHost, ByteCount(128)}};
                 auto tracker =
                     std::make_unique<MemoryTracker>(MemoryTracker::Create(limits).value());
                 CpuAllocator allocator(*tracker);
                 Buffer buffer = allocator.Allocate(HostRequest(64, MemoryCategory::kTest)).value();
                 tracker.reset();
                 static_cast<void>(buffer);
               }()),
               "");
}

TEST(M2LifetimeDeathTest, PoolDestructionRejectsLiveLeases) {
  EXPECT_DEATH(([] {
                 CpuAllocator allocator;
                 Buffer backing =
                     allocator.Allocate(HostRequest(64, MemoryCategory::kRuntimeInternal)).value();
                 auto pool = std::make_unique<FixedBufferPool>(
                     FixedBufferPool::Create(
                         std::move(backing),
                         PoolGeometry{1, ByteCount(64), ByteCount(64), PoolGeneration(0)})
                         .value());
                 BufferLease lease = pool->Acquire().value();
                 pool.reset();
                 static_cast<void>(lease);
               }()),
               "");
}

TEST(M2WorkspaceTest, CheckedBumpAllocationDoesNotAdvanceOnFailure) {
  CpuAllocator allocator;
  Buffer backing = allocator.Allocate(HostRequest(128, MemoryCategory::kWorkspace)).value();
  WorkspaceArenaPool pool =
      WorkspaceArenaPool::Create(std::move(backing), 2, ByteCount(64), ByteCount(32)).value();
  WorkspaceLease lease = pool.Acquire().value();
  EXPECT_TRUE(lease.Allocate(ByteCount(7), ByteCount(1), WorkspaceTag::kTest).ok());
  EXPECT_TRUE(lease.Allocate(ByteCount(8), ByteCount(8), WorkspaceTag::kTest).ok());
  EXPECT_EQ(lease.used().value(), 16);
  EXPECT_EQ(lease.Allocate(ByteCount(60), ByteCount(1), WorkspaceTag::kTest).status().code(),
            absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(lease.used().value(), 16);
  EXPECT_TRUE(lease.Release().ok());
  EXPECT_TRUE(pool.Close().ok());
}

TEST(M2CompletionFenceTest, PendingCompleteAcknowledgeAndStale) {
  testing::FakeFenceDomain domain(1);
  CompletionFence fence = domain.Acquire().value();
  const FenceToken token = fence.token();
  EXPECT_EQ(fence.Poll()->state, FenceState::kPending);
  EXPECT_EQ(fence.Acknowledge().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(domain.Complete(token).ok());
  EXPECT_EQ(fence.Poll()->state, FenceState::kComplete);
  EXPECT_TRUE(fence.Acknowledge().ok());
  EXPECT_EQ(domain.Poll(token).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(M2CompletionFenceTest, FailureAndAbandonAreDeferred) {
  testing::FakeFenceDomain domain(1);
  FenceToken token{Device::Host(), FenceSlotId(0), FenceGeneration(0)};
  {
    CompletionFence fence = domain.Acquire().value();
    token = fence.token();
  }
  EXPECT_EQ(domain.abandoned_count(), 1);
  EXPECT_TRUE(domain.Fail(token, absl::UnavailableError("injected")).ok());
  EXPECT_TRUE(domain.ReclaimAbandoned().ok());
  EXPECT_EQ(domain.abandoned_count(), 0);
  EXPECT_TRUE(domain.Acquire().ok());
}

TEST(M2CompletionFenceTest, PendingWaitTimesOutWithoutRecycling) {
  testing::FakeFenceDomain domain(1);
  CompletionFence fence = domain.Acquire().value();
  EXPECT_EQ(fence.WaitUntil(Deadline(Nanoseconds(0)), FenceWaitReason::kTest).code(),
            absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(fence.Acknowledge().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(domain.Acquire().status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_TRUE(domain.Complete(fence.token()).ok());
  EXPECT_EQ(fence.Poll()->state, FenceState::kComplete);
  EXPECT_TRUE(fence.Acknowledge().ok());
}

}  // namespace
}  // namespace inferx
