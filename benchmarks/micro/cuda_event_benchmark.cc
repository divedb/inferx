#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "inferx/base/clock.h"
#include "inferx/platform/cuda/cuda_device.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/cuda_event_pool.h"
#include "inferx/platform/cuda/cuda_stream.h"

namespace {

#if defined(__has_attribute)
#if __has_attribute(noipa)
#define INFERX_BENCHMARK_NOIPA __attribute__((noipa))
#elif __has_attribute(noinline)
#define INFERX_BENCHMARK_NOIPA __attribute__((noinline))
#else
#define INFERX_BENCHMARK_NOIPA
#endif
#elif defined(__GNUC__)
#define INFERX_BENCHMARK_NOIPA __attribute__((noipa))
#else
#define INFERX_BENCHMARK_NOIPA
#endif

enum class DirectEventState : uint8_t {
  kFree,
  kLeased,
  kRecorded,
  kDeferred,
  kRetired,
};

struct DirectEventSlot {
  cudaEvent_t event = nullptr;
  uint64_t generation = 0;
  DirectEventState state = DirectEventState::kFree;
  bool terminal_observed = false;
};

struct DirectEventStream {
  cudaStream_t handle = nullptr;
  inferx::DeviceId device = inferx::DeviceId(0);
};

class DirectEventLease;

class DirectEventDomain final : public inferx::FenceDomain {
 public:
  DirectEventDomain(inferx::DeviceId device, std::array<DirectEventSlot, 8>& slots,
                    inferx::cuda::CudaHealth* health) noexcept
      : device_(device), slots_(slots), health_(health) {}

  [[nodiscard]] INFERX_BENCHMARK_NOIPA absl::StatusOr<DirectEventLease> Acquire();

  [[nodiscard]] INFERX_BENCHMARK_NOIPA absl::Status Record(inferx::FenceToken token,
                                                           const DirectEventStream& stream) {
    if (health_ != nullptr) {
      absl::Status accepting = health_->CheckAcceptingWork();
      if (!accepting.ok()) return accepting;
    }
    absl::StatusOr<cudaEvent_t> event = Resolve(token, false);
    if (!event.ok()) return event.status();
    if (stream.device != device_) {
      return absl::InvalidArgumentError("benchmark.event: stream and event devices differ");
    }
    DirectEventSlot& slot = slots_[token.slot.value()];
    if (slot.state == DirectEventState::kLeased) {
      absl::Status status = inferx::cuda::CudaErrorStatus(
          cudaEventRecord(*event, stream.handle), "benchmark-event-record", device_, health_);
      if (status.ok()) slot.state = DirectEventState::kRecorded;
      return status;
    }
    return absl::FailedPreconditionError("benchmark.event: direct record state mismatch");
  }

  [[nodiscard]] INFERX_BENCHMARK_NOIPA absl::Status WaitOn(inferx::FenceToken token,
                                                           const DirectEventStream& stream) {
    if (health_ != nullptr) {
      absl::Status accepting = health_->CheckAcceptingWork();
      if (!accepting.ok()) return accepting;
    }
    absl::StatusOr<cudaEvent_t> event = Resolve(token, true);
    if (!event.ok()) return event.status();
    if (stream.device != device_) {
      return absl::InvalidArgumentError("benchmark.event: stream and event devices differ");
    }
    return inferx::cuda::CudaErrorStatus(cudaStreamWaitEvent(stream.handle, *event, 0),
                                         "benchmark-stream-wait-event", device_, health_);
  }

  [[nodiscard]] INFERX_BENCHMARK_NOIPA absl::StatusOr<inferx::FencePoll> Poll(
      inferx::FenceToken token) override {
    absl::StatusOr<cudaEvent_t> event = Resolve(token, true);
    if (!event.ok()) return event.status();
    const cudaError_t error = cudaEventQuery(*event);
    if (error == cudaSuccess) {
      slots_[token.slot.value()].terminal_observed = true;
      return inferx::FencePoll{inferx::FenceState::kComplete, absl::OkStatus()};
    }
    if (error == cudaErrorNotReady) {
      return inferx::FencePoll{inferx::FenceState::kPending, absl::OkStatus()};
    }
    absl::Status status =
        inferx::cuda::CudaErrorStatus(error, "benchmark-event-query", device_, health_);
    slots_[token.slot.value()].terminal_observed = true;
    return inferx::FencePoll{inferx::FenceState::kFailed, status};
  }

  [[nodiscard]] absl::Status WaitUntil(inferx::FenceToken, inferx::Deadline,
                                       inferx::FenceWaitReason) override {
    return absl::UnimplementedError("benchmark.event: direct wait is not measured");
  }

  [[nodiscard]] INFERX_BENCHMARK_NOIPA absl::Status Acknowledge(inferx::FenceToken token) override {
    absl::StatusOr<cudaEvent_t> event = Resolve(token, true);
    if (!event.ok()) return event.status();
    static_cast<void>(event);
    DirectEventSlot& slot = slots_[token.slot.value()];
    if (!slot.terminal_observed) {
      return absl::FailedPreconditionError("benchmark.event: terminal state was not observed");
    }
    slot.state = DirectEventState::kFree;
    slot.terminal_observed = false;
    return absl::OkStatus();
  }

  void Abandon(inferx::FenceToken token) noexcept override {
    if (token.device != inferx::Device::Cuda(device_) || token.slot.value() >= slots_.size())
      return;
    DirectEventSlot& slot = slots_[token.slot.value()];
    if (slot.generation != token.generation.value()) return;
    if (slot.state == DirectEventState::kRecorded) slot.state = DirectEventState::kDeferred;
    if (slot.state == DirectEventState::kLeased) slot.state = DirectEventState::kFree;
  }

 private:
  [[nodiscard]] INFERX_BENCHMARK_NOIPA absl::StatusOr<cudaEvent_t> Resolve(inferx::FenceToken token,
                                                                           bool require_recorded) {
    if (token.device != inferx::Device::Cuda(device_) || token.slot.value() >= slots_.size()) {
      return absl::FailedPreconditionError("benchmark.event: stale direct token");
    }
    DirectEventSlot& slot = slots_[token.slot.value()];
    if (slot.generation != token.generation.value() || slot.state == DirectEventState::kFree ||
        slot.state == DirectEventState::kRetired ||
        (require_recorded && slot.state != DirectEventState::kRecorded &&
         slot.state != DirectEventState::kDeferred)) {
      return absl::FailedPreconditionError("benchmark.event: stale direct token");
    }
    return slot.event;
  }

  inferx::DeviceId device_;
  std::array<DirectEventSlot, 8>& slots_;
  inferx::cuda::CudaHealth* health_;
  bool closed_ = false;
};

class DirectEventLease {
 public:
  INFERX_BENCHMARK_NOIPA DirectEventLease(DirectEventDomain& domain,
                                          inferx::FenceToken token) noexcept
      : domain_(&domain), token_(token) {}
  INFERX_BENCHMARK_NOIPA ~DirectEventLease() noexcept {
    if (domain_ != nullptr) domain_->Abandon(token_);
  }

  DirectEventLease(const DirectEventLease&) = delete;
  DirectEventLease& operator=(const DirectEventLease&) = delete;
  INFERX_BENCHMARK_NOIPA DirectEventLease(DirectEventLease&& other) noexcept
      : domain_(std::exchange(other.domain_, nullptr)),
        token_(other.token_),
        recorded_(std::exchange(other.recorded_, false)) {}
  INFERX_BENCHMARK_NOIPA DirectEventLease& operator=(DirectEventLease&& other) noexcept {
    if (this != &other) {
      if (domain_ != nullptr) domain_->Abandon(token_);
      domain_ = std::exchange(other.domain_, nullptr);
      token_ = other.token_;
      recorded_ = std::exchange(other.recorded_, false);
    }
    return *this;
  }

  [[nodiscard]] INFERX_BENCHMARK_NOIPA absl::Status Record(const DirectEventStream& stream) {
    if (domain_ == nullptr) {
      return absl::FailedPreconditionError("benchmark.event: direct lease is inactive");
    }
    absl::Status status = domain_->Record(token_, stream);
    if (status.ok()) recorded_ = true;
    return status;
  }

  [[nodiscard]] INFERX_BENCHMARK_NOIPA absl::Status WaitOn(const DirectEventStream& stream) {
    if (domain_ == nullptr || !recorded_) {
      return absl::FailedPreconditionError("benchmark.event: direct lease is not recorded");
    }
    return domain_->WaitOn(token_, stream);
  }

  [[nodiscard]] INFERX_BENCHMARK_NOIPA absl::StatusOr<inferx::CompletionFence> IntoFence() {
    if (domain_ == nullptr || !recorded_) {
      return absl::FailedPreconditionError("benchmark.event: direct lease is not recorded");
    }
    DirectEventDomain* domain = std::exchange(domain_, nullptr);
    recorded_ = false;
    return inferx::CompletionFence(domain, token_);
  }

 private:
  DirectEventDomain* domain_ = nullptr;
  inferx::FenceToken token_{inferx::Device::Host(), inferx::FenceSlotId(0),
                            inferx::FenceGeneration(0)};
  bool recorded_ = false;
};

absl::StatusOr<DirectEventLease> DirectEventDomain::Acquire() {
  if (closed_) {
    return absl::FailedPreconditionError("benchmark.event: direct pool is closed");
  }
  if (health_ != nullptr) {
    absl::Status accepting = health_->CheckAcceptingWork();
    if (!accepting.ok()) return accepting;
  }
  for (size_t index = 0; index < slots_.size(); ++index) {
    DirectEventSlot& slot = slots_[index];
    if (slot.state != DirectEventState::kFree) continue;
    if (slot.generation == std::numeric_limits<uint64_t>::max()) {
      slot.state = DirectEventState::kRetired;
      continue;
    }
    ++slot.generation;
    slot.state = DirectEventState::kLeased;
    slot.terminal_observed = false;
    return DirectEventLease(*this,
                            inferx::FenceToken{inferx::Device::Cuda(device_),
                                               inferx::FenceSlotId(static_cast<uint32_t>(index)),
                                               inferx::FenceGeneration(slot.generation)});
  }
  return absl::ResourceExhaustedError("benchmark.event: direct pool exhausted");
}

void BM_CudaEventRecordWaitAcknowledge(benchmark::State& state) {
  auto devices = inferx::cuda::DiscoverCudaDevices();
  if (!devices.ok() || devices->empty()) {
    state.SkipWithError("no CUDA device");
    return;
  }
  const inferx::DeviceId device = devices->front().ordinal;
  inferx::cuda::CudaHealth health;
  auto guard = inferx::cuda::CudaDeviceGuard::Create(device);
  if (!guard.ok()) {
    state.SkipWithError("CUDA device guard failed");
    return;
  }
  auto producer = inferx::cuda::CudaStream::Create(device, inferx::cuda::CudaStreamRole::kCompute,
                                                   0, inferx::cuda::CudaApi::Production(), &health);
  auto consumer = inferx::cuda::CudaStream::Create(device, inferx::cuda::CudaStreamRole::kTransfer,
                                                   0, inferx::cuda::CudaApi::Production(), &health);
  auto pool =
      inferx::cuda::CudaEventPool::Create(device, 8, inferx::cuda::CudaApi::Production(), &health);
  if (!producer.ok() || !consumer.ok() || !pool.ok()) {
    state.SkipWithError("CUDA setup failed");
    if (pool.ok()) (*pool)->Close().IgnoreError();
    if (consumer.ok()) consumer->Close().IgnoreError();
    if (producer.ok()) producer->Close().IgnoreError();
    guard->Restore().IgnoreError();
    return;
  }
  int64_t completed_iterations = 0;
  int64_t query_calls = 0;
  for (auto _ : state) {
    static_cast<void>(_);
    auto event = (*pool)->Acquire().value();
    if (!event.Record(*producer).ok() || !event.WaitOn(*consumer).ok()) {
      state.SkipWithError("wrapper event submission failed");
      break;
    }
    auto fence = event.IntoFence().value();
    bool poll_failed = false;
    while (true) {
      auto poll = fence.Poll();
      ++query_calls;
      if (!poll.ok()) {
        state.SkipWithError("wrapper event query failed");
        poll_failed = true;
        break;
      }
      if (poll->state != inferx::FenceState::kPending) break;
    }
    if (poll_failed) break;
    if (!fence.Acknowledge().ok()) {
      state.SkipWithError("wrapper event acknowledgement failed");
      break;
    }
    ++completed_iterations;
  }
  state.SetItemsProcessed(completed_iterations);
  state.counters["cuda_event_record_calls"] = static_cast<double>(completed_iterations);
  state.counters["cuda_event_query_calls"] = static_cast<double>(query_calls);
  state.counters["cuda_stream_wait_event_calls"] = static_cast<double>(completed_iterations);
  (*pool)->Close().IgnoreError();
  consumer->Close().IgnoreError();
  producer->Close().IgnoreError();
  guard->Restore().IgnoreError();
}

void BM_CudaEventRecordWaitQueryDirect(benchmark::State& state) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    state.SkipWithError("no CUDA device");
    return;
  }
  int ordinal = 0;
  if (cudaGetDevice(&ordinal) != cudaSuccess || ordinal < 0) {
    state.SkipWithError("direct CUDA device lookup failed");
    return;
  }
  const inferx::DeviceId device(static_cast<uint32_t>(ordinal));
  DirectEventStream producer{nullptr, device};
  DirectEventStream consumer{nullptr, device};
  std::array<DirectEventSlot, 8> slots{};
  if (cudaStreamCreateWithFlags(&producer.handle, cudaStreamNonBlocking) != cudaSuccess ||
      cudaStreamCreateWithFlags(&consumer.handle, cudaStreamNonBlocking) != cudaSuccess) {
    state.SkipWithError("direct CUDA event setup failed");
    if (consumer.handle != nullptr) cudaStreamDestroy(consumer.handle);
    if (producer.handle != nullptr) cudaStreamDestroy(producer.handle);
    return;
  }
  bool events_created = true;
  for (DirectEventSlot& slot : slots) {
    if (cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming) != cudaSuccess) {
      events_created = false;
      break;
    }
  }
  if (!events_created) {
    state.SkipWithError("direct CUDA event setup failed");
    for (DirectEventSlot& slot : slots) {
      if (slot.event != nullptr) cudaEventDestroy(slot.event);
    }
    cudaStreamDestroy(consumer.handle);
    cudaStreamDestroy(producer.handle);
    return;
  }
  inferx::cuda::CudaHealth health;
  DirectEventDomain domain(device, slots, &health);
  int64_t completed_iterations = 0;
  int64_t query_calls = 0;
  for (auto _ : state) {
    static_cast<void>(_);
    absl::StatusOr<DirectEventLease> event_result = domain.Acquire();
    if (!event_result.ok()) {
      state.SkipWithError("direct event pool exhausted");
      break;
    }
    DirectEventLease event = std::move(*event_result);
    if (!event.Record(producer).ok()) {
      state.SkipWithError("direct event record failed");
      break;
    }
    if (!event.WaitOn(consumer).ok()) {
      state.SkipWithError("direct event wait failed");
      break;
    }
    absl::StatusOr<inferx::CompletionFence> fence_result = event.IntoFence();
    if (!fence_result.ok()) {
      state.SkipWithError("direct event fence conversion failed");
      break;
    }
    inferx::CompletionFence fence = std::move(*fence_result);
    bool poll_failed = false;
    while (true) {
      absl::StatusOr<inferx::FencePoll> poll = fence.Poll();
      ++query_calls;
      if (!poll.ok()) {
        state.SkipWithError("direct event query failed");
        poll_failed = true;
        break;
      }
      if (poll->state != inferx::FenceState::kPending) break;
    }
    if (poll_failed) break;
    if (!fence.Acknowledge().ok()) {
      state.SkipWithError("direct event acknowledgement failed");
      break;
    }
    ++completed_iterations;
  }
  state.SetItemsProcessed(completed_iterations);
  state.counters["cuda_event_record_calls"] = static_cast<double>(completed_iterations);
  state.counters["cuda_event_query_calls"] = static_cast<double>(query_calls);
  state.counters["cuda_stream_wait_event_calls"] = static_cast<double>(completed_iterations);
  for (DirectEventSlot& slot : slots) {
    cudaEventDestroy(slot.event);
  }
  cudaStreamDestroy(consumer.handle);
  cudaStreamDestroy(producer.handle);
}

BENCHMARK(BM_CudaEventRecordWaitAcknowledge)->Repetitions(30)->MinWarmUpTime(0.1)->UseRealTime();
BENCHMARK(BM_CudaEventRecordWaitQueryDirect)->Repetitions(30)->MinWarmUpTime(0.1)->UseRealTime();

#undef INFERX_BENCHMARK_NOIPA

}  // namespace

BENCHMARK_MAIN();
