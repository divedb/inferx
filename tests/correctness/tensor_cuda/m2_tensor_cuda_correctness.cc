#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <system_error>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/base/clock.h"
#include "inferx/platform/cuda/cuda_allocator.h"
#include "inferx/platform/cuda/cuda_buffer_access.h"
#include "inferx/platform/cuda/cuda_copy.h"
#include "inferx/platform/cuda/cuda_device.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/cuda_event_pool.h"
#include "inferx/platform/cuda/cuda_stream.h"
#include "inferx/runtime/memory_tracker.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"
#include "inferx/tensor/tensor_view.h"
#include "test_kernels.h"

namespace inferx::cuda {
namespace {

constexpr uint64_t kBufferBytes = 65536;

TEST(M2TensorCudaCorrectnessTest, DeterministicStridedCasesMatchCpu) {
  const auto devices = DiscoverCudaDevices();
  ASSERT_TRUE(devices.ok()) << devices.status();
  ASSERT_FALSE(devices->empty());
  const CudaDeviceInfo& info = devices->front();
  const CudaApi& api = CudaApi::Production();
  CudaHealth health;
  auto guard = CudaDeviceGuard::Create(info.ordinal, api, &health);
  ASSERT_TRUE(guard.ok()) << guard.status();
  const std::array<MemoryLimit, 2> limits{
      MemoryLimit{Device::Host(), MemoryKind::kPinnedHost, ByteCount(4 * kBufferBytes)},
      MemoryLimit{Device::Cuda(info.ordinal), MemoryKind::kDevice, ByteCount(4 * kBufferBytes)}};
  MemoryTracker tracker = MemoryTracker::Create(limits).value();
  CudaPinnedAllocator pinned(info.ordinal, tracker, health, api);
  CudaDeviceAllocator device(info.ordinal, tracker, health, api);
  CpuAllocator cpu;
  const AllocationRequest pinned_request{Device::Host(), MemoryKind::kPinnedHost,
                                         ByteCount(kBufferBytes), ByteCount(64),
                                         MemoryCategory::kTest};
  const AllocationRequest device_request{Device::Cuda(info.ordinal), MemoryKind::kDevice,
                                         ByteCount(kBufferBytes), ByteCount(64),
                                         MemoryCategory::kTest};
  const AllocationRequest host_request{Device::Host(), MemoryKind::kHost, ByteCount(kBufferBytes),
                                       ByteCount(64), MemoryCategory::kTest};
  Buffer pinned_input = pinned.Allocate(pinned_request).value();
  Buffer pinned_output = pinned.Allocate(pinned_request).value();
  Buffer device_input = device.Allocate(device_request).value();
  Buffer device_output = device.Allocate(device_request).value();
  Buffer expected = cpu.Allocate(host_request).value();
  MutableBufferView pinned_input_view =
      pinned_input.MutableView({ByteCount(0), ByteCount(kBufferBytes)}).value();
  MutableBufferView pinned_output_view =
      pinned_output.MutableView({ByteCount(0), ByteCount(kBufferBytes)}).value();
  MutableBufferView device_input_view =
      device_input.MutableView({ByteCount(0), ByteCount(kBufferBytes)}).value();
  MutableBufferView device_output_view =
      device_output.MutableView({ByteCount(0), ByteCount(kBufferBytes)}).value();
  MutableBufferView expected_view =
      expected.MutableView({ByteCount(0), ByteCount(kBufferBytes)}).value();
  CudaStream transfer =
      CudaStream::Create(info.ordinal, CudaStreamRole::kTransfer, 0, api, &health).value();
  CudaStream compute =
      CudaStream::Create(info.ordinal, CudaStreamRole::kCompute, 0, api, &health).value();
  std::unique_ptr<CudaEventPool> events =
      std::move(CudaEventPool::Create(info.ordinal, 3, api, &health).value());

  constexpr std::array<DType, 4> kDTypes{DType::kUInt8, DType::kUInt16, DType::kFloat32,
                                         DType::kFloat64};
  uint64_t case_count = 1000;
  if (const char* raw = std::getenv("INFERX_M2_CUDA_CORRECTNESS_CASES")) {
    const char* end = raw + std::strlen(raw);
    const std::from_chars_result parsed = std::from_chars(raw, end, case_count);
    ASSERT_EQ(parsed.ec, std::errc{});
    ASSERT_EQ(parsed.ptr, end);
    ASSERT_GE(case_count, 1000);
  }
  for (uint64_t test_case = 0; test_case < case_count; ++test_case) {
    const uint8_t rank = static_cast<uint8_t>(test_case % 9);
    std::array<uint64_t, kMaxTensorRank> dimensions{};
    for (uint8_t axis = 0; axis < rank; ++axis) {
      dimensions[axis] = 1 + ((test_case + axis * 7) % 2);
    }
    if (rank > 0 && test_case % 127 == 0) dimensions[test_case % rank] = 0;
    const Shape shape = Shape::Create(std::span<const uint64_t>(dimensions.data(), rank)).value();
    const DType dtype = kDTypes[test_case % kDTypes.size()];
    const uint64_t element_size = DTypeSize(dtype)->value();
    const auto make_strides = [&](bool row_major, uint64_t padding) {
      std::array<uint64_t, kMaxTensorRank> values{};
      uint64_t stride = 1;
      for (uint8_t index = 0; index < rank; ++index) {
        const uint8_t axis = row_major ? static_cast<uint8_t>(rank - index - 1) : index;
        values[axis] = stride;
        const uint64_t extent = dimensions[axis] == 0 ? 1 : dimensions[axis];
        stride *= extent;
        if (index == 0 && rank > 1) stride += padding;
      }
      return values;
    };
    const std::array<uint64_t, kMaxTensorRank> source_values =
        make_strides((test_case & 1U) == 0, test_case % 3);
    const std::array<uint64_t, kMaxTensorRank> destination_values =
        make_strides((test_case & 2U) == 0, (test_case / 3) % 3);
    const Strides source_strides =
        Strides::CreateElements(std::span<const uint64_t>(source_values.data(), rank)).value();
    const Strides destination_strides =
        Strides::CreateElements(std::span<const uint64_t>(destination_values.data(), rank)).value();
    const LayoutAnalysis source_layout = AnalyzeLayout(shape, source_strides, dtype).value();
    const LayoutAnalysis destination_layout =
        AnalyzeLayout(shape, destination_strides, dtype).value();
    const uint64_t source_byte_offset = (test_case % 4) * element_size;
    const uint64_t destination_byte_offset = ((test_case / 4) % 4) * element_size;
    ASSERT_LE(source_layout.reachable_bytes.value() + source_byte_offset, kBufferBytes);
    ASSERT_LE(destination_layout.reachable_bytes.value() + destination_byte_offset, kBufferBytes);
    for (uint64_t index = 0; index < kBufferBytes; ++index) {
      (*pinned_input_view.HostBytes())[index] =
          static_cast<std::byte>((index * 17 + test_case * 13) & 0xff);
    }
    std::memset(pinned_output_view.HostBytes()->data(), 0, kBufferBytes);
    std::memset(expected_view.HostBytes()->data(), 0, kBufferBytes);
    const TensorView cpu_source = TensorView::Create(pinned_input_view.AsConst(), dtype, shape,
                                                     source_strides, ByteCount(source_byte_offset))
                                      .value();
    const MutableTensorView cpu_destination =
        MutableTensorView::Create(expected_view, dtype, shape, destination_strides,
                                  ByteCount(destination_byte_offset))
            .value();
    ASSERT_TRUE(CopyTensorCpu(cpu_source, cpu_destination).ok());

    ASSERT_TRUE(CopyAsync(CopyRequest{pinned_input_view.AsConst(), device_input_view,
                                      ByteCount(source_byte_offset +
                                                source_layout.reachable_bytes.value())},
                          transfer, api, &health)
                    .ok());
    CudaEventLease uploaded = events->Acquire().value();
    ASSERT_TRUE(uploaded.Record(transfer).ok());
    ASSERT_TRUE(uploaded.WaitOn(compute).ok());
    ASSERT_TRUE(MemsetAsync(device_output_view, 0, compute, api, &health).ok());
    test::StridedCopyParams params;
    params.rank = rank;
    params.element_size = static_cast<uint32_t>(element_size);
    params.element_count = shape.NumElements().value();
    params.source_byte_offset = source_byte_offset;
    params.destination_byte_offset = destination_byte_offset;
    std::copy_n(dimensions.begin(), rank, params.dimensions);
    std::copy_n(source_values.begin(), rank, params.source_strides);
    std::copy_n(destination_values.begin(), rank, params.destination_strides);
    test::LaunchStridedCopy(BufferAccess::Address(device_input_view.AsConst()),
                            BufferAccess::Address(device_output_view), params, compute.handle());
    ASSERT_EQ(api.peek_at_last_error(), cudaSuccess);
    CudaEventLease computed = events->Acquire().value();
    ASSERT_TRUE(computed.Record(compute).ok());
    ASSERT_TRUE(computed.WaitOn(transfer).ok());
    ASSERT_TRUE(CopyAsync(CopyRequest{device_output_view.AsConst(), pinned_output_view,
                                      ByteCount(destination_byte_offset +
                                                destination_layout.reachable_bytes.value())},
                          transfer, api, &health)
                    .ok());
    CudaEventLease final_event = events->Acquire().value();
    ASSERT_TRUE(final_event.Record(transfer).ok());
    CompletionFence fence = final_event.IntoFence().value();
    const Deadline deadline =
        std::chrono::time_point_cast<Nanoseconds>(std::chrono::steady_clock::now()) +
        std::chrono::seconds(5);
    ASSERT_TRUE(fence.WaitUntil(deadline, FenceWaitReason::kTest).ok());
    ASSERT_EQ(fence.Poll()->state, FenceState::kComplete);
    ASSERT_TRUE(fence.Acknowledge().ok());
    ASSERT_TRUE(uploaded.Release().ok());
    ASSERT_TRUE(computed.Release().ok());
    const size_t result_bytes =
        static_cast<size_t>(destination_byte_offset + destination_layout.reachable_bytes.value());
    EXPECT_TRUE(
        std::equal(expected_view.HostBytes()->begin(),
                   expected_view.HostBytes()->begin() + static_cast<std::ptrdiff_t>(result_bytes),
                   pinned_output_view.HostBytes()->begin()));
  }

  EXPECT_TRUE(events->Close().ok());
  EXPECT_TRUE(compute.Close().ok());
  EXPECT_TRUE(transfer.Close().ok());
  EXPECT_TRUE(device_output.Release().ok());
  EXPECT_TRUE(device_input.Release().ok());
  EXPECT_TRUE(pinned_output.Release().ok());
  EXPECT_TRUE(pinned_input.Release().ok());
  EXPECT_TRUE(expected.Release().ok());
  EXPECT_TRUE(tracker.ValidateBaseline().ok());
  EXPECT_TRUE(guard->Restore().ok());
}

}  // namespace
}  // namespace inferx::cuda
