#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "inferx/base/status.h"
#include "inferx/platform/cuda/cuda_copy.h"
#include "inferx/platform/cuda/cuda_device.h"
#include "inferx/platform/cuda/cuda_device_context.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/cuda_event_pool.h"
#include "inferx/platform/cuda/cuda_stream.h"
#include "inferx/tensor/buffer.h"
#include "tests/fakes/fake_allocator.h"

namespace inferx::cuda {
namespace {

struct FakeCudaState {
  int current_device = 0;
  int set_device_calls = 0;
  int stream_create_calls = 0;
  int stream_destroy_calls = 0;
  int event_create_calls = 0;
  int event_destroy_calls = 0;
  int event_record_calls = 0;
  int event_query_calls = 0;
  int wait_calls = 0;
  int pointer_attribute_calls = 0;
  int copy_calls = 0;
  int allocation_calls = 0;
  int successful_allocations = 0;
  int free_calls = 0;
  int device_sync_calls = 0;
  int fail_stream_create_at = 0;
  int fail_event_create_at = 0;
  int fail_allocation_at = 0;
  int fail_set_device_at = 0;
  bool fail_stream_destroy = false;
  bool fail_event_destroy = false;
  bool fail_free = false;
  bool return_null_allocation = false;
  bool event_recorded = false;
  unsigned int stream_flags = 0;
  unsigned int event_flags = 0;
  alignas(16) std::array<std::byte, 16> stream_storage{};
  alignas(16) std::array<std::byte, 16> event_storage{};
};

FakeCudaState* g_fake = nullptr;

cudaError_t FakeGetDeviceCount(int* count) {
  *count = 1;
  return cudaSuccess;
}
cudaError_t FakeGetDevice(int* device) {
  *device = g_fake->current_device;
  return cudaSuccess;
}
cudaError_t FakeSetDevice(int device) {
  const int ordinal = ++g_fake->set_device_calls;
  if (g_fake->fail_set_device_at == ordinal) return cudaErrorUnknown;
  g_fake->current_device = device;
  return cudaSuccess;
}
cudaError_t FakeGetDeviceProperties(cudaDeviceProp* properties, int) {
  *properties = cudaDeviceProp{};
  std::strncpy(properties->name, "fake-sm89", sizeof(properties->name) - 1);
  properties->major = 8;
  properties->minor = 9;
  properties->totalGlobalMem = 8ULL * 1024 * 1024 * 1024;
  properties->multiProcessorCount = 76;
  properties->warpSize = 32;
  properties->maxThreadsPerBlock = 1024;
  properties->asyncEngineCount = 2;
  properties->unifiedAddressing = 1;
  properties->canMapHostMemory = 1;
  properties->managedMemory = 1;
  properties->streamPrioritiesSupported = 1;
  properties->memoryPoolsSupported = 1;
  return cudaSuccess;
}
cudaError_t FakeDriverVersion(int* version) {
  *version = 12080;
  return cudaSuccess;
}
cudaError_t FakeRuntimeVersion(int* version) {
  *version = 12080;
  return cudaSuccess;
}
cudaError_t FakePriorityRange(int* least, int* greatest) {
  *least = 0;
  *greatest = -2;
  return cudaSuccess;
}
cudaError_t FakeStreamCreate(cudaStream_t* stream, unsigned int flags, int) {
  const int ordinal = ++g_fake->stream_create_calls;
  if (g_fake->fail_stream_create_at == ordinal) return cudaErrorMemoryAllocation;
  *stream = reinterpret_cast<cudaStream_t>(g_fake->stream_storage.data());
  g_fake->stream_flags = flags;
  return cudaSuccess;
}
cudaError_t FakeStreamDestroy(cudaStream_t) {
  ++g_fake->stream_destroy_calls;
  return g_fake->fail_stream_destroy ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t FakeStreamWait(cudaStream_t, cudaEvent_t, unsigned int) {
  ++g_fake->wait_calls;
  return cudaSuccess;
}
cudaError_t FakeEventCreate(cudaEvent_t* event, unsigned int flags) {
  const int ordinal = ++g_fake->event_create_calls;
  if (g_fake->fail_event_create_at == ordinal) return cudaErrorMemoryAllocation;
  *event = reinterpret_cast<cudaEvent_t>(g_fake->event_storage.data());
  g_fake->event_flags = flags;
  return cudaSuccess;
}
cudaError_t FakeEventDestroy(cudaEvent_t) {
  ++g_fake->event_destroy_calls;
  return g_fake->fail_event_destroy ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t FakeEventRecord(cudaEvent_t, cudaStream_t) {
  g_fake->event_recorded = true;
  ++g_fake->event_record_calls;
  return cudaSuccess;
}
cudaError_t FakeEventQuery(cudaEvent_t) {
  ++g_fake->event_query_calls;
  return g_fake->event_recorded ? cudaSuccess : cudaErrorNotReady;
}
cudaError_t FakePointerAttributes(cudaPointerAttributes* attributes, const void*) {
  attributes->type =
      g_fake->pointer_attribute_calls++ == 0 ? cudaMemoryTypeHost : cudaMemoryTypeDevice;
  return cudaSuccess;
}
cudaError_t FakeMemcpy(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) {
  ++g_fake->copy_calls;
  return cudaSuccess;
}
cudaError_t FakeAllocate(void** address, size_t bytes) {
  const int ordinal = ++g_fake->allocation_calls;
  if (g_fake->fail_allocation_at == ordinal) return cudaErrorMemoryAllocation;
  if (g_fake->return_null_allocation) {
    *address = nullptr;
    ++g_fake->successful_allocations;
    return cudaSuccess;
  }
  constexpr size_t kAlignment = 256;
  const size_t rounded = (bytes + kAlignment - 1) & ~(kAlignment - 1);
  *address = std::aligned_alloc(kAlignment, rounded);
  if (*address == nullptr) return cudaErrorMemoryAllocation;
  ++g_fake->successful_allocations;
  return cudaSuccess;
}
cudaError_t FakeHostAllocate(void** address, size_t bytes, unsigned int) {
  return FakeAllocate(address, bytes);
}
cudaError_t FakeFree(void* address) {
  if (g_fake->fail_free) {
    ++g_fake->free_calls;
    return cudaErrorUnknown;
  }
  std::free(address);
  ++g_fake->free_calls;
  return cudaSuccess;
}
cudaError_t FakeDeviceSynchronize() {
  ++g_fake->device_sync_calls;
  return cudaSuccess;
}

CudaApi MakeFakeApi(FakeCudaState& state) {
  g_fake = &state;
  CudaApi api = CudaApi::Production();
  api.get_device_count = &FakeGetDeviceCount;
  api.get_device = &FakeGetDevice;
  api.set_device = &FakeSetDevice;
  api.get_device_properties = &FakeGetDeviceProperties;
  api.driver_get_version = &FakeDriverVersion;
  api.runtime_get_version = &FakeRuntimeVersion;
  api.device_get_stream_priority_range = &FakePriorityRange;
  api.stream_create_with_priority = &FakeStreamCreate;
  api.stream_destroy = &FakeStreamDestroy;
  api.stream_wait_event = &FakeStreamWait;
  api.event_create_with_flags = &FakeEventCreate;
  api.event_destroy = &FakeEventDestroy;
  api.event_record = &FakeEventRecord;
  api.event_query = &FakeEventQuery;
  api.pointer_get_attributes = &FakePointerAttributes;
  api.memcpy_async = &FakeMemcpy;
  api.malloc_device = &FakeAllocate;
  api.free_device = &FakeFree;
  api.host_alloc = &FakeHostAllocate;
  api.free_host = &FakeFree;
  api.device_synchronize = &FakeDeviceSynchronize;
  return api;
}

CudaContextConfig FakeContextConfig() {
  CudaContextConfig config;
  config.device = DeviceId(0);
  config.accepted_sm = 89;
  config.device_reserve = ByteCount(0);
  config.device_budget = ByteCount(4ULL * 1024 * 1024);
  config.pinned_budget = ByteCount(4ULL * 1024 * 1024);
  config.event_pool_slots = 8;
  config.metadata_ring_slots = 2;
  config.metadata_slot_bytes = ByteCount(4096);
  config.staging_pool_slots = 2;
  config.staging_slot_bytes = ByteCount(4096);
  config.workspace_slots = 1;
  config.workspace_bytes_per_slot = ByteCount(1024 * 1024);
  return config;
}

MemoryTracker FakeContextTracker() {
  const std::array<MemoryLimit, 2> limits{
      MemoryLimit{Device::Host(), MemoryKind::kPinnedHost, ByteCount(4ULL * 1024 * 1024)},
      MemoryLimit{Device::Cuda(DeviceId(0)), MemoryKind::kDevice, ByteCount(4ULL * 1024 * 1024)}};
  return MemoryTracker::Create(limits).value();
}

class CountingDeferredResource final : public CudaDeferredResource {
 public:
  explicit CountingDeferredResource(int* polls) : polls_(polls) {}

  absl::StatusOr<bool> TryReclaim() override {
    ++*polls_;
    return *polls_ >= 2;
  }

 private:
  int* polls_;
};

TEST(M2CudaUnitTest, DiscoveryCapabilityAndNestedGuardRestore) {
  FakeCudaState state;
  const CudaApi api = MakeFakeApi(state);
  const auto devices = DiscoverCudaDevices(api);
  ASSERT_TRUE(devices.ok());
  ASSERT_EQ(devices->size(), 1);
  EXPECT_EQ(devices->front().compute_major, 8);
  EXPECT_EQ(devices->front().compute_minor, 9);
  constexpr std::array<int, 1> kAccepted{89};
  EXPECT_TRUE(
      ValidateCudaCapabilities(devices->front(), kAccepted, ByteCount(1024), ByteCount(4096)).ok());
  constexpr std::array<int, 1> kRejected{90};
  EXPECT_EQ(
      ValidateCudaCapabilities(devices->front(), kRejected, ByteCount(0), ByteCount(0)).code(),
      absl::StatusCode::kUnimplemented);
  CudaDeviceInfo old_runtime = devices->front();
  old_runtime.runtime_version = 12070;
  EXPECT_EQ(ValidateCudaCapabilities(old_runtime, kAccepted, ByteCount(0), ByteCount(0)).code(),
            absl::StatusCode::kUnimplemented);

  CudaDeviceGuard outer = CudaDeviceGuard::Create(DeviceId(1), api).value();
  EXPECT_EQ(state.current_device, 1);
  CudaDeviceGuard inner = CudaDeviceGuard::Create(DeviceId(2), api).value();
  EXPECT_EQ(state.current_device, 2);
  EXPECT_TRUE(inner.Restore().ok());
  EXPECT_EQ(state.current_device, 1);
  EXPECT_TRUE(outer.Restore().ok());
  EXPECT_EQ(state.current_device, 0);
}

TEST(M2CudaUnitTest, ErrorMappingPoisonsOnlyStickyFaults) {
  CudaHealth health;
  absl::Status invalid = CudaErrorStatus(cudaErrorInvalidDevice, "test", DeviceId(7), &health);
  EXPECT_EQ(invalid.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(GetErrorReason(invalid).value(), ErrorReason::kCudaInvalidDevice);
  EXPECT_EQ(health.state(), CudaHealthState::kHealthy);

  absl::Status sticky = CudaErrorStatus(cudaErrorIllegalAddress, "test", DeviceId(7), &health);
  EXPECT_EQ(sticky.code(), absl::StatusCode::kUnavailable);
  EXPECT_EQ(GetErrorReason(sticky).value(), ErrorReason::kCudaAsyncFault);
  EXPECT_EQ(health.state(), CudaHealthState::kPoisoned);
  EXPECT_EQ(health.poison_cause()->message(), sticky.message());

  CudaHealth invariant_health;
  absl::Status invariant =
      CudaErrorStatus(cudaErrorInvalidResourceHandle, "test", DeviceId(7), &invariant_health);
  EXPECT_EQ(invariant.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(GetErrorReason(invariant).value(), ErrorReason::kInvariantViolation);
  EXPECT_EQ(invariant_health.state(), CudaHealthState::kPoisoned);

  CudaHealth lost_health;
  absl::Status lost =
      CudaErrorStatus(cudaErrorContextIsDestroyed, "test", DeviceId(7), &lost_health);
  EXPECT_EQ(lost.code(), absl::StatusCode::kUnavailable);
  EXPECT_EQ(GetErrorReason(lost).value(), ErrorReason::kCudaDeviceLost);
  EXPECT_EQ(lost_health.state(), CudaHealthState::kPoisoned);

  CudaHealth timeout_health;
  absl::Status timeout =
      CudaErrorStatus(cudaErrorLaunchTimeout, "test", DeviceId(7), &timeout_health);
  EXPECT_EQ(timeout.code(), absl::StatusCode::kUnavailable);
  EXPECT_EQ(GetErrorReason(timeout).value(), ErrorReason::kCudaAsyncFault);
  EXPECT_EQ(timeout_health.state(), CudaHealthState::kPoisoned);
}

TEST(M2CudaUnitTest, NonblockingStreamAndGenerationCheckedEventLifecycle) {
  FakeCudaState state;
  const CudaApi api = MakeFakeApi(state);
  CudaHealth health;
  CudaStream stream =
      CudaStream::Create(DeviceId(0), CudaStreamRole::kCompute, -1, api, &health).value();
  EXPECT_EQ(state.stream_flags, cudaStreamNonBlocking);
  EXPECT_EQ(stream.priority(), -1);
  std::unique_ptr<CudaEventPool> pool =
      std::move(CudaEventPool::Create(DeviceId(0), 1, api, &health).value());
  EXPECT_EQ(state.event_flags, cudaEventDisableTiming);
  CudaEventLease event = pool->Acquire().value();
  const FenceToken token = event.token();
  EXPECT_EQ(event.Poll().status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(event.Record(stream).ok());
  EXPECT_TRUE(event.WaitOn(stream).ok());
  CompletionFence fence = event.IntoFence().value();
  EXPECT_EQ(fence.Poll()->state, FenceState::kComplete);
  EXPECT_TRUE(fence.Acknowledge().ok());
  EXPECT_EQ(pool->Poll(token).status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(pool->Close().ok());
  EXPECT_TRUE(stream.Close().ok());
  EXPECT_EQ(state.event_create_calls, state.event_destroy_calls);
  EXPECT_EQ(state.stream_create_calls, state.stream_destroy_calls);
}

TEST(M2CudaUnitTest, EventGenerationWrapRetiresTheSlotWithoutOutstandingWork) {
  FakeCudaState state;
  const CudaApi api = MakeFakeApi(state);
  CudaHealth health;
  CudaStream stream =
      CudaStream::Create(DeviceId(0), CudaStreamRole::kCompute, 0, api, &health).value();
  std::unique_ptr<CudaEventPool> pool =
      std::move(CudaEventPool::Create(DeviceId(0), 1, api, &health,
                                      FenceGeneration(std::numeric_limits<uint64_t>::max() - 1))
                    .value());
  CudaEventLease final_generation = pool->Acquire().value();
  EXPECT_TRUE(final_generation.Record(stream).ok());
  EXPECT_EQ(final_generation.Poll()->state, FenceState::kComplete);
  EXPECT_TRUE(final_generation.Release().ok());
  EXPECT_EQ(pool->Acquire().status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(pool->available_slots(), 0);
  EXPECT_FALSE(pool->has_outstanding_events());
  EXPECT_TRUE(pool->Close().ok());
  EXPECT_TRUE(stream.Close().ok());
}

TEST(M2CudaUnitTest, MetadataGenerationWrapRetiresSlotsAndReleasesBacking) {
  FakeCudaState state;
  const CudaApi api = MakeFakeApi(state);
  CudaHealth health;
  MemoryTracker tracker = FakeContextTracker();
  CudaPinnedAllocator pinned(DeviceId(0), tracker, health, api);
  CudaDeviceAllocator device(DeviceId(0), tracker, health, api);
  std::unique_ptr<CudaEventPool> events =
      std::move(CudaEventPool::Create(DeviceId(0), 8, api, &health).value());
  CudaMetadataRing ring =
      CudaMetadataRing::Create(2, ByteCount(4096), pinned, device, *events, DeviceId(0), api,
                               &health, PoolGeneration(std::numeric_limits<uint64_t>::max()))
          .value();
  EXPECT_EQ(ring.Acquire().status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_TRUE(ring.ValidateInvariants().ok());
  EXPECT_TRUE(ring.Close().ok());
  EXPECT_TRUE(events->Close().ok());
  EXPECT_EQ(state.successful_allocations, state.free_calls);
  EXPECT_TRUE(tracker.ValidateBaseline().ok());
}

TEST(M2CudaUnitTest, ContextCreationClosesEveryInjectedFailurePrefix) {
  struct FailurePoint {
    int stream = 0;
    int event = 0;
    int allocation = 0;
  };
  constexpr std::array<FailurePoint, 9> kFailures{
      FailurePoint{1, 0, 0}, FailurePoint{2, 0, 0}, FailurePoint{0, 3, 0},
      FailurePoint{0, 0, 1}, FailurePoint{0, 0, 2}, FailurePoint{0, 0, 3},
      FailurePoint{0, 0, 4}, FailurePoint{0, 0, 5}, FailurePoint{0, 0, 6}};

  for (const FailurePoint failure : kFailures) {
    FakeCudaState state;
    state.fail_stream_create_at = failure.stream;
    state.fail_event_create_at = failure.event;
    state.fail_allocation_at = failure.allocation;
    const CudaApi api = MakeFakeApi(state);
    MemoryTracker tracker = FakeContextTracker();
    const auto context = CudaDeviceContext::Create(FakeContextConfig(), tracker, api);
    EXPECT_FALSE(context.ok());
    EXPECT_EQ(state.stream_destroy_calls,
              state.stream_create_calls - (failure.stream == 0 ? 0 : 1));
    EXPECT_EQ(state.event_destroy_calls, state.event_create_calls - (failure.event == 0 ? 0 : 1));
    EXPECT_EQ(state.free_calls, state.successful_allocations);
    EXPECT_TRUE(tracker.ValidateBaseline().ok());
  }
}

TEST(M2CudaUnitTest, ContextSuccessfulShutdownClosesTheCompletePrefix) {
  FakeCudaState state;
  const CudaApi api = MakeFakeApi(state);
  MemoryTracker tracker = FakeContextTracker();
  auto context = CudaDeviceContext::Create(FakeContextConfig(), tracker, api);
  ASSERT_TRUE(context.ok()) << context.status();
  EXPECT_TRUE((*context)->Shutdown(Deadline(Nanoseconds::max())).ok());
  EXPECT_EQ(state.stream_create_calls, state.stream_destroy_calls);
  EXPECT_EQ(state.event_create_calls, state.event_destroy_calls);
  EXPECT_EQ(state.successful_allocations, state.free_calls);
  EXPECT_EQ(state.device_sync_calls, 1);
  EXPECT_TRUE(tracker.ValidateBaseline().ok());
}

TEST(M2CudaUnitTest, ContextOwnsDeferredResourcesUntilTheyReportReclaimed) {
  FakeCudaState state;
  const CudaApi api = MakeFakeApi(state);
  MemoryTracker tracker = FakeContextTracker();
  auto context = CudaDeviceContext::Create(FakeContextConfig(), tracker, api);
  ASSERT_TRUE(context.ok()) << context.status();
  int polls = 0;
  EXPECT_TRUE((*context)->DeferResource(std::make_unique<CountingDeferredResource>(&polls)).ok());
  EXPECT_EQ((*context)->deferred_resource_count(), 1);
  EXPECT_TRUE((*context)->ReclaimDeferredResources().ok());
  EXPECT_EQ((*context)->deferred_resource_count(), 1);
  EXPECT_TRUE((*context)->ReclaimDeferredResources().ok());
  EXPECT_EQ((*context)->deferred_resource_count(), 0);
  EXPECT_TRUE((*context)->Shutdown(Deadline(Nanoseconds::max())).ok());
  EXPECT_TRUE(tracker.ValidateBaseline().ok());
}

TEST(M2CudaUnitTest, DestroyFailuresPoisonResourceOwnership) {
  FakeCudaState stream_state;
  stream_state.fail_stream_destroy = true;
  const CudaApi stream_api = MakeFakeApi(stream_state);
  CudaHealth stream_health;
  CudaStream stream =
      CudaStream::Create(DeviceId(0), CudaStreamRole::kCompute, 0, stream_api, &stream_health)
          .value();
  EXPECT_FALSE(stream.Close().ok());
  EXPECT_EQ(stream_health.state(), CudaHealthState::kPoisoned);
  EXPECT_TRUE(stream_health.poison_cause().has_value());

  FakeCudaState event_state;
  event_state.fail_event_destroy = true;
  const CudaApi event_api = MakeFakeApi(event_state);
  CudaHealth event_health;
  std::unique_ptr<CudaEventPool> pool =
      std::move(CudaEventPool::Create(DeviceId(0), 1, event_api, &event_health).value());
  EXPECT_FALSE(pool->Close().ok());
  EXPECT_EQ(event_health.state(), CudaHealthState::kPoisoned);
  EXPECT_TRUE(event_health.poison_cause().has_value());
  const int destroy_calls = event_state.event_destroy_calls;
  EXPECT_FALSE(pool->Close().ok());
  EXPECT_EQ(event_state.event_destroy_calls, destroy_calls);
}

TEST(M2CudaUnitTest, StreamCreationRestoreFailureDestroysTheCreatedStream) {
  FakeCudaState state;
  state.fail_set_device_at = 2;
  const CudaApi api = MakeFakeApi(state);
  CudaHealth health;
  const auto stream = CudaStream::Create(DeviceId(1), CudaStreamRole::kCompute, 0, api, &health);
  EXPECT_FALSE(stream.ok());
  EXPECT_EQ(state.stream_create_calls, 1);
  EXPECT_EQ(state.stream_destroy_calls, 1);
  EXPECT_EQ(health.state(), CudaHealthState::kPoisoned);
}

TEST(M2CudaUnitTest, EventCreationPreservesPrimaryFailureAndAnnotatesCleanup) {
  FakeCudaState state;
  state.fail_event_create_at = 2;
  state.fail_event_destroy = true;
  const CudaApi api = MakeFakeApi(state);
  CudaHealth health;
  const auto pool = CudaEventPool::Create(DeviceId(0), 3, api, &health);
  EXPECT_FALSE(pool.ok());
  EXPECT_EQ(pool.status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_NE(pool.status().message().find("cleanup failure"), absl::string_view::npos);
  EXPECT_EQ(state.event_create_calls, 2);
  EXPECT_EQ(state.event_destroy_calls, 1);
  EXPECT_EQ(health.state(), CudaHealthState::kPoisoned);
}

TEST(M2CudaUnitTest, FailedCleanupFreePoisonsAndAccountsLeakedBytes) {
  FakeCudaState state;
  state.return_null_allocation = true;
  state.fail_free = true;
  const CudaApi api = MakeFakeApi(state);
  CudaHealth health;
  MemoryTracker tracker = FakeContextTracker();
  CudaDeviceAllocator allocator(DeviceId(0), tracker, health, api);
  const absl::Status result = allocator
                                  .Allocate({Device::Cuda(DeviceId(0)), MemoryKind::kDevice,
                                             ByteCount(4096), ByteCount(64), MemoryCategory::kTest})
                                  .status();
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(result.message().find("cleanup failure"), absl::string_view::npos);
  EXPECT_EQ(health.state(), CudaHealthState::kPoisoned);
  const auto snapshot = tracker.Snapshot();
  ASSERT_EQ(snapshot.size(), 1);
  EXPECT_EQ(snapshot.front().second.leaked.value(), 4096);
  EXPECT_EQ(snapshot.front().second.live_allocations, 1);
  EXPECT_FALSE(tracker.ValidateBaseline().ok());
  EXPECT_TRUE(tracker.ValidateBaseline(true).ok());
}

TEST(M2CudaLifetimeDeathTest, ContextDestructionRequiresExplicitShutdown) {
  EXPECT_DEATH(([] {
                 FakeCudaState state;
                 const CudaApi api = MakeFakeApi(state);
                 MemoryTracker tracker = FakeContextTracker();
                 auto context = CudaDeviceContext::Create(FakeContextConfig(), tracker, api);
                 if (!context.ok()) std::abort();
               }()),
               "");
}

#if !defined(NDEBUG) || defined(INFERX_CUDA_DEBUG_POINTERS)
TEST(M2CudaUnitTest, CopyChecksPointerKindsAndRejectsPageableMemory) {
  FakeCudaState state;
  const CudaApi api = MakeFakeApi(state);
  CudaHealth health;
  CudaStream stream =
      CudaStream::Create(DeviceId(0), CudaStreamRole::kTransfer, 0, api, &health).value();
  testing::FakeAllocator allocator(ByteCount(256));
  Buffer pinned = allocator
                      .Allocate({Device::Host(), MemoryKind::kPinnedHost, ByteCount(64),
                                 ByteCount(16), MemoryCategory::kTest})
                      .value();
  Buffer device = allocator
                      .Allocate({Device::Cuda(DeviceId(0)), MemoryKind::kDevice, ByteCount(64),
                                 ByteCount(16), MemoryCategory::kTest})
                      .value();
  Buffer pageable = allocator
                        .Allocate({Device::Host(), MemoryKind::kHost, ByteCount(64), ByteCount(16),
                                   MemoryCategory::kTest})
                        .value();
  MutableBufferView device_view = device.MutableView({ByteCount(0), ByteCount(64)}).value();
  EXPECT_TRUE(
      CopyAsync({pinned.View({ByteCount(0), ByteCount(64)}).value(), device_view, ByteCount(64)},
                stream, api, &health)
          .ok());
  EXPECT_EQ(state.pointer_attribute_calls, 2);
  EXPECT_EQ(state.copy_calls, 1);
  EXPECT_EQ(
      CopyAsync({pageable.View({ByteCount(0), ByteCount(64)}).value(), device_view, ByteCount(64)},
                stream, api, &health)
          .code(),
      absl::StatusCode::kFailedPrecondition);
  const int pointer_calls_before_poison = state.pointer_attribute_calls;
  health.Poison(absl::UnavailableError("test poison"));
  EXPECT_EQ(MemsetAsync(device_view, 0, stream, api, &health).code(),
            absl::StatusCode::kUnavailable);
  EXPECT_EQ(state.pointer_attribute_calls, pointer_calls_before_poison);
  EXPECT_TRUE(pageable.Release().ok());
  EXPECT_TRUE(device.Release().ok());
  EXPECT_TRUE(pinned.Release().ok());
  EXPECT_TRUE(stream.Close().ok());
}
#endif

}  // namespace
}  // namespace inferx::cuda
