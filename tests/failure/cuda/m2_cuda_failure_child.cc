#include <cuda_runtime_api.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/clock.h"
#include "inferx/base/status.h"
#include "inferx/platform/cuda/cuda_api.h"
#include "inferx/platform/cuda/cuda_device.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/cuda_event_pool.h"
#include "inferx/platform/cuda/cuda_stream.h"
#include "test_kernels.h"

namespace {

inferx::Deadline DeadlineAfter(std::chrono::seconds duration) {
  return std::chrono::time_point_cast<inferx::Nanoseconds>(std::chrono::steady_clock::now()) +
         duration;
}

int RunStickyFault(std::string_view mode) {
  const auto devices = inferx::cuda::DiscoverCudaDevices();
  if (!devices.ok()) {
    std::cerr << devices.status() << '\n';
    return 2;
  }
  const inferx::DeviceId device = devices->front().ordinal;
  inferx::cuda::CudaHealth health;
  auto stream = inferx::cuda::CudaStream::Create(device, inferx::cuda::CudaStreamRole::kCompute, 0,
                                                 inferx::cuda::CudaApi::Production(), &health);
  if (!stream.ok()) {
    std::cerr << stream.status() << '\n';
    return 3;
  }
  auto pool =
      inferx::cuda::CudaEventPool::Create(device, 1, inferx::cuda::CudaApi::Production(), &health);
  if (!pool.ok()) {
    std::cerr << pool.status() << '\n';
    return 4;
  }
  if (mode == "--illegal-address") {
    inferx::cuda::test::LaunchIllegalAddress(stream->handle());
  } else if (mode == "--assert") {
    inferx::cuda::test::LaunchDeviceAssert(stream->handle());
  } else {
    std::cerr << "unknown fault mode\n";
    return 5;
  }
  const cudaError_t launch = inferx::cuda::CudaApi::Production().peek_at_last_error();
  if (launch != cudaSuccess) {
    std::cerr << "fault kernel was rejected synchronously: " << launch << '\n';
    return 6;
  }
  auto event = (*pool)->Acquire();
  if (!event.ok() || !event->Record(*stream).ok()) {
    std::cerr << "could not record fault observation event\n";
    return 7;
  }
  auto fence = event->IntoFence();
  if (!fence.ok()) {
    std::cerr << fence.status() << '\n';
    return 8;
  }
  const absl::Status observed =
      fence->WaitUntil(DeadlineAfter(std::chrono::seconds(10)), inferx::FenceWaitReason::kTest);
  const absl::StatusOr<inferx::ErrorReason> reason = inferx::GetErrorReason(observed);
  if (observed.code() != absl::StatusCode::kUnavailable || !reason.ok() ||
      *reason != inferx::ErrorReason::kCudaAsyncFault ||
      health.state() != inferx::cuda::CudaHealthState::kPoisoned ||
      health.CheckAcceptingWork().ok()) {
    std::cerr << "sticky fault classification failed: " << observed << '\n';
    return 9;
  }
  std::cout << "expected isolated CUDA fault classified and context poisoned\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: inferx_cuda_failure_child --illegal-address|--assert\n";
    return 1;
  }
  return RunStickyFault(argv[1]);
}
