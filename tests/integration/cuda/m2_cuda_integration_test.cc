#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "inferx/base/clock.h"
#include "inferx/base/token.h"
#include "inferx/platform/cuda/cuda_api.h"
#include "inferx/platform/cuda/cuda_device.h"
#include "inferx/platform/cuda/cuda_device_context.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/cuda_event_pool.h"
#include "inferx/runtime/memory_tracker.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/device.h"
#include "test_kernels.h"
#include "tests/integration/cuda/cuda_test_pipeline.h"

namespace inferx::cuda::testing {
namespace {

constexpr uint64_t kBytes = 4096;
constexpr uint64_t kBudget = 64ULL * 1024 * 1024;
constexpr int kAcceptedSm = 89;

Deadline DeadlineAfter(std::chrono::seconds duration) {
  return std::chrono::time_point_cast<Nanoseconds>(std::chrono::steady_clock::now()) + duration;
}

struct OwnedContext {
  MemoryTracker tracker;
  std::unique_ptr<CudaDeviceContext> context;
};

absl::StatusOr<std::unique_ptr<OwnedContext>> CreateContext(DeviceId device,
                                                            bool transfer_stream = true) {
  const MemoryLimit limits[] = {{Device::Host(), MemoryKind::kPinnedHost, ByteCount(kBudget)},
                                {Device::Cuda(device), MemoryKind::kDevice, ByteCount(kBudget)}};
  absl::StatusOr<MemoryTracker> tracker = MemoryTracker::Create(limits);
  if (!tracker.ok()) return tracker.status();
  CudaContextConfig config;
  config.device = device;
  config.accepted_sm = kAcceptedSm;
  config.device_reserve = ByteCount(0);
  config.device_budget = ByteCount(kBudget);
  config.pinned_budget = ByteCount(kBudget);
  config.event_pool_slots = 8;
  config.metadata_ring_slots = 2;
  config.metadata_slot_bytes = ByteCount(kBytes);
  config.staging_pool_slots = 2;
  config.staging_slot_bytes = ByteCount(kBytes);
  config.workspace_slots = 1;
  config.workspace_bytes_per_slot = ByteCount(1024 * 1024);
  config.enable_transfer_stream = transfer_stream;
  try {
    auto owned = std::make_unique<OwnedContext>(OwnedContext{std::move(*tracker), nullptr});
    absl::StatusOr<std::unique_ptr<CudaDeviceContext>> context =
        CudaDeviceContext::Create(config, owned->tracker);
    if (!context.ok()) return context.status();
    owned->context = std::move(*context);
    return owned;
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("cuda.integration: host context fixture allocation failed");
  }
}

std::vector<std::byte> InputPattern() {
  std::vector<std::byte> input(kBytes);
  for (uint64_t index = 0; index < kBytes; ++index) {
    input[index] = static_cast<std::byte>((index * 37U + 11U) & 0xffU);
  }
  return input;
}

test::StridedCopyParams ContiguousParameters() {
  test::StridedCopyParams parameters;
  parameters.dimensions[0] = kBytes;
  parameters.source_strides[0] = 1;
  parameters.destination_strides[0] = 1;
  parameters.element_count = kBytes;
  parameters.element_size = 1;
  parameters.rank = 1;
  return parameters;
}

test::StridedCopyParams StridedParameters() {
  test::StridedCopyParams parameters;
  parameters.dimensions[0] = 7;
  parameters.dimensions[1] = 5;
  parameters.source_strides[0] = 9;
  parameters.source_strides[1] = 1;
  parameters.destination_strides[0] = 1;
  parameters.destination_strides[1] = 8;
  parameters.source_byte_offset = 4;
  parameters.destination_byte_offset = 6;
  parameters.element_count = 35;
  parameters.element_size = 2;
  parameters.rank = 2;
  return parameters;
}

absl::Status WaitForTerminal(CudaTestSubmission& submission) {
  const Deadline deadline = DeadlineAfter(std::chrono::seconds(10));
  while (true) {
    absl::StatusOr<FencePoll> poll = submission.Poll();
    if (!poll.ok()) return poll.status();
    if (poll->state == FenceState::kComplete) return absl::OkStatus();
    if (poll->state == FenceState::kFailed) return poll->completion_status;
    if (std::chrono::steady_clock::now() >= deadline) {
      return absl::DeadlineExceededError(
          "cuda.integration: submission completion deadline expired");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

void RunRoundTrip(bool transfer_stream) {
  const auto devices = DiscoverCudaDevices();
  ASSERT_TRUE(devices.ok()) << devices.status();
  auto owned = CreateContext(devices->front().ordinal, transfer_stream);
  ASSERT_TRUE(owned.ok()) << owned.status();
  CudaTestPipeline pipeline(*(*owned)->context);
  const std::vector<std::byte> input = InputPattern();
  auto submission = pipeline.Submit(input, ContiguousParameters());
  ASSERT_TRUE(submission.ok()) << submission.status();
  ASSERT_TRUE(WaitForTerminal(*submission).ok());
  auto output = submission->FinishCompleted();
  ASSERT_TRUE(output.ok()) << output.status();
  auto output_bytes = output->view().HostBytes();
  ASSERT_TRUE(output_bytes.ok()) << output_bytes.status();
  EXPECT_TRUE(std::equal(input.begin(), input.end(), output_bytes->begin()));
  EXPECT_TRUE(output->Release().ok());
  EXPECT_TRUE(pipeline.Close().ok());
  EXPECT_TRUE((*owned)->context->Shutdown(DeadlineAfter(std::chrono::seconds(10))).ok());
  EXPECT_TRUE((*owned)->tracker.ValidateBaseline().ok());
}

TEST(M2CudaIntegrationTest, TransferStreamPipelineCompletesBeforeOutputAccess) {
  RunRoundTrip(true);
}

TEST(M2CudaIntegrationTest, SingleStreamPipelineHasIdenticalOutput) { RunRoundTrip(false); }

TEST(M2CudaIntegrationTest, StridedPipelineUsesRetainedWorkspaceViews) {
  const auto devices = DiscoverCudaDevices();
  ASSERT_TRUE(devices.ok()) << devices.status();
  auto owned = CreateContext(devices->front().ordinal);
  ASSERT_TRUE(owned.ok()) << owned.status();
  CudaTestPipeline pipeline(*(*owned)->context);
  const std::vector<std::byte> input = InputPattern();
  const test::StridedCopyParams parameters = StridedParameters();
  std::vector<std::byte> expected(kBytes, std::byte{0});
  for (uint64_t row = 0; row < parameters.dimensions[0]; ++row) {
    for (uint64_t column = 0; column < parameters.dimensions[1]; ++column) {
      const uint64_t source =
          parameters.source_byte_offset +
          (row * parameters.source_strides[0] + column * parameters.source_strides[1]) *
              parameters.element_size;
      const uint64_t destination =
          parameters.destination_byte_offset +
          (row * parameters.destination_strides[0] + column * parameters.destination_strides[1]) *
              parameters.element_size;
      std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(source), parameters.element_size,
                  expected.begin() + static_cast<std::ptrdiff_t>(destination));
    }
  }
  auto submission = pipeline.Submit(input, parameters);
  ASSERT_TRUE(submission.ok()) << submission.status();
  ASSERT_TRUE(WaitForTerminal(*submission).ok());
  auto output = submission->FinishCompleted();
  ASSERT_TRUE(output.ok()) << output.status();
  auto output_bytes = output->view().HostBytes();
  ASSERT_TRUE(output_bytes.ok()) << output_bytes.status();
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), output_bytes->begin()));
  EXPECT_TRUE(output->Release().ok());
  EXPECT_TRUE(pipeline.Close().ok());
  EXPECT_TRUE((*owned)->context->Shutdown(DeadlineAfter(std::chrono::seconds(10))).ok());
  EXPECT_TRUE((*owned)->tracker.ValidateBaseline().ok());
}

TEST(M2CudaIntegrationTest, ExhaustionAcknowledgementAndReuseAreDeterministic) {
  const auto devices = DiscoverCudaDevices();
  ASSERT_TRUE(devices.ok()) << devices.status();
  auto owned = CreateContext(devices->front().ordinal);
  ASSERT_TRUE(owned.ok()) << owned.status();
  CudaTestPipeline pipeline(*(*owned)->context);
  const std::vector<std::byte> input = InputPattern();
  auto first = pipeline.Submit(input, ContiguousParameters());
  ASSERT_TRUE(first.ok()) << first.status();
  const auto exhausted = pipeline.Submit(input, ContiguousParameters());
  EXPECT_EQ(exhausted.status().code(), absl::StatusCode::kResourceExhausted);
  ASSERT_TRUE(WaitForTerminal(*first).ok());
  auto first_output = first->FinishCompleted();
  ASSERT_TRUE(first_output.ok()) << first_output.status();
  EXPECT_TRUE(first_output->Release().ok());

  auto reused = pipeline.Submit(input, ContiguousParameters());
  ASSERT_TRUE(reused.ok()) << reused.status();
  ASSERT_TRUE(WaitForTerminal(*reused).ok());
  auto reused_output = reused->FinishCompleted();
  ASSERT_TRUE(reused_output.ok()) << reused_output.status();
  EXPECT_TRUE(reused_output->Release().ok());
  EXPECT_TRUE(pipeline.Close().ok());
  EXPECT_TRUE((*owned)->context->Shutdown(DeadlineAfter(std::chrono::seconds(10))).ok());
}

TEST(M2CudaIntegrationTest, PendingDestructionDefersTheWholeResourceBundle) {
  const auto devices = DiscoverCudaDevices();
  ASSERT_TRUE(devices.ok()) << devices.status();
  auto owned = CreateContext(devices->front().ordinal);
  ASSERT_TRUE(owned.ok()) << owned.status();
  CudaTestPipeline pipeline(*(*owned)->context);
  {
    auto submission = pipeline.Submit(InputPattern(), ContiguousParameters());
    ASSERT_TRUE(submission.ok()) << submission.status();
  }
  EXPECT_EQ(pipeline.deferred_count(), 1);
  const Deadline deadline = DeadlineAfter(std::chrono::seconds(10));
  while (pipeline.deferred_count() != 0 && std::chrono::steady_clock::now() < deadline) {
    ASSERT_TRUE(pipeline.ReclaimDeferred().ok());
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  EXPECT_EQ(pipeline.deferred_count(), 0);
  EXPECT_TRUE(pipeline.Close().ok());
  EXPECT_TRUE((*owned)->context->Shutdown(DeadlineAfter(std::chrono::seconds(10))).ok());
}

TEST(M2CudaIntegrationTest, PipelineDestructionTransfersPendingBundleToContext) {
  const auto devices = DiscoverCudaDevices();
  ASSERT_TRUE(devices.ok()) << devices.status();
  auto owned = CreateContext(devices->front().ordinal);
  ASSERT_TRUE(owned.ok()) << owned.status();
  test::LaunchBoundedDelay(100000000, (*owned)->context->compute_stream().handle());
  ASSERT_EQ((*owned)->context->api().peek_at_last_error(), cudaSuccess);
  {
    CudaTestPipeline pipeline(*(*owned)->context);
    auto submission = pipeline.Submit(InputPattern(), ContiguousParameters());
    ASSERT_TRUE(submission.ok()) << submission.status();
  }
  EXPECT_EQ((*owned)->context->deferred_resource_count(), 1);
  EXPECT_TRUE((*owned)->context->Shutdown(DeadlineAfter(std::chrono::seconds(10))).ok());
  EXPECT_EQ((*owned)->context->deferred_resource_count(), 0);
  EXPECT_TRUE((*owned)->tracker.ValidateBaseline().ok());
}

TEST(M2CudaIntegrationTest, ShutdownRetainsCompletedUnacknowledgedWorkForRetry) {
  const auto devices = DiscoverCudaDevices();
  ASSERT_TRUE(devices.ok()) << devices.status();
  auto owned = CreateContext(devices->front().ordinal);
  ASSERT_TRUE(owned.ok()) << owned.status();
  auto event = (*owned)->context->event_pool().Acquire();
  ASSERT_TRUE(event.ok()) << event.status();
  ASSERT_TRUE(event->Record((*owned)->context->compute_stream()).ok());
  auto fence = event->IntoFence();
  ASSERT_TRUE(fence.ok()) << fence.status();
  ASSERT_TRUE(
      fence->WaitUntil(DeadlineAfter(std::chrono::seconds(10)), FenceWaitReason::kTest).ok());
  EXPECT_EQ(
      (*owned)
          ->context
          ->Shutdown(std::chrono::time_point_cast<Nanoseconds>(std::chrono::steady_clock::now()))
          .code(),
      absl::StatusCode::kDeadlineExceeded);
  ASSERT_EQ(fence->Poll()->state, FenceState::kComplete);
  EXPECT_TRUE(fence->Acknowledge().ok());
  EXPECT_TRUE((*owned)->context->Shutdown(DeadlineAfter(std::chrono::seconds(10))).ok());
}

TEST(M2CudaIntegrationTest, PendingBoundedDelayPreventsEarlyShutdownAndCanRetry) {
  const auto devices = DiscoverCudaDevices();
  ASSERT_TRUE(devices.ok()) << devices.status();
  auto owned = CreateContext(devices->front().ordinal);
  ASSERT_TRUE(owned.ok()) << owned.status();
  test::LaunchBoundedDelay(100000000, (*owned)->context->compute_stream().handle());
  ASSERT_EQ(CudaApi::Production().peek_at_last_error(), cudaSuccess);
  auto event = (*owned)->context->event_pool().Acquire();
  ASSERT_TRUE(event.ok()) << event.status();
  ASSERT_TRUE(event->Record((*owned)->context->compute_stream()).ok());
  auto fence = event->IntoFence();
  ASSERT_TRUE(fence.ok()) << fence.status();
  EXPECT_EQ(
      (*owned)
          ->context
          ->Shutdown(std::chrono::time_point_cast<Nanoseconds>(std::chrono::steady_clock::now()))
          .code(),
      absl::StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(
      fence->WaitUntil(DeadlineAfter(std::chrono::seconds(10)), FenceWaitReason::kTest).ok());
  ASSERT_EQ(fence->Poll()->state, FenceState::kComplete);
  EXPECT_TRUE(fence->Acknowledge().ok());
  EXPECT_TRUE((*owned)->context->Shutdown(DeadlineAfter(std::chrono::seconds(10))).ok());
}

TEST(M2CudaIntegrationTest, MultipleVisibleDevicesRemainContextIsolated) {
  const auto devices = DiscoverCudaDevices();
  ASSERT_TRUE(devices.ok()) << devices.status();
  if (devices->size() < 2) return;
  auto first = CreateContext((*devices)[0].ordinal);
  ASSERT_TRUE(first.ok()) << first.status();
  auto second = CreateContext((*devices)[1].ordinal);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_NE((*first)->context->info().ordinal, (*second)->context->info().ordinal);
  EXPECT_TRUE((*second)->context->Shutdown(DeadlineAfter(std::chrono::seconds(10))).ok());
  EXPECT_TRUE((*first)->context->Shutdown(DeadlineAfter(std::chrono::seconds(10))).ok());
}

}  // namespace
}  // namespace inferx::cuda::testing
