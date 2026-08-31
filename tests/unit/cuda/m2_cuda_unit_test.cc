#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/base/status.h"
#include "inferx/platform/cuda/cuda_copy.h"
#include "inferx/platform/cuda/cuda_device.h"
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
  g_fake->current_device = device;
  ++g_fake->set_device_calls;
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
  *stream = reinterpret_cast<cudaStream_t>(g_fake->stream_storage.data());
  g_fake->stream_flags = flags;
  ++g_fake->stream_create_calls;
  return cudaSuccess;
}
cudaError_t FakeStreamDestroy(cudaStream_t) {
  ++g_fake->stream_destroy_calls;
  return cudaSuccess;
}
cudaError_t FakeStreamWait(cudaStream_t, cudaEvent_t, unsigned int) {
  ++g_fake->wait_calls;
  return cudaSuccess;
}
cudaError_t FakeEventCreate(cudaEvent_t* event, unsigned int flags) {
  *event = reinterpret_cast<cudaEvent_t>(g_fake->event_storage.data());
  g_fake->event_flags = flags;
  ++g_fake->event_create_calls;
  return cudaSuccess;
}
cudaError_t FakeEventDestroy(cudaEvent_t) {
  ++g_fake->event_destroy_calls;
  return cudaSuccess;
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
  return api;
}

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
  EXPECT_TRUE(pageable.Release().ok());
  EXPECT_TRUE(device.Release().ok());
  EXPECT_TRUE(pinned.Release().ok());
  EXPECT_TRUE(stream.Close().ok());
}
#endif

}  // namespace
}  // namespace inferx::cuda
