#include "tests/integration/cuda/cuda_test_pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"
#include "inferx/platform/cuda/cuda_api.h"
#include "inferx/platform/cuda/cuda_buffer_access.h"
#include "inferx/platform/cuda/cuda_copy.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/tensor/buffer.h"

namespace inferx::cuda::testing {
namespace {

absl::Status PendingSubmission(absl::string_view operation) {
  return WithErrorReason(absl::FailedPreconditionError(absl::StrCat(
                             "cuda.test_submission.", operation, ": completion remains pending")),
                         ErrorReason::kPendingResource);
}

}  // namespace

struct CudaTestSubmission::Bundle {
  CudaDeviceContext* context = nullptr;
  BufferLease input;
  BufferLease output;
  CudaMetadataLease metadata;
  WorkspaceLease workspace;
  CudaEventLease compute_done;
  CompletionFence completion;
  bool queued = false;
};

struct CudaTestSubmission::State {
  explicit State(CudaDeviceContext& supplied_context) : context(&supplied_context) {}

  CudaDeviceContext* context;
  mutable std::mutex mutex;
  std::vector<std::unique_ptr<Bundle>> deferred;
  bool closed = false;
};

namespace {

absl::Status ReleaseBundle(CudaTestSubmission::Bundle& bundle, bool require_complete) {
  if (bundle.completion.active()) {
    absl::StatusOr<FencePoll> poll = bundle.completion.Poll();
    if (!poll.ok()) return poll.status();
    if (poll->state == FenceState::kPending) {
      return require_complete ? PendingSubmission("finish") : absl::OkStatus();
    }
    absl::Status acknowledged = bundle.completion.Acknowledge();
    if (!acknowledged.ok()) return acknowledged;
  }
  if (bundle.compute_done.recorded()) {
    absl::Status released = bundle.compute_done.Release();
    if (!released.ok()) return released;
  } else {
    absl::Status released = bundle.compute_done.Release();
    if (!released.ok() && released.code() != absl::StatusCode::kFailedPrecondition) {
      return released;
    }
  }
  absl::Status metadata = bundle.metadata.Release();
  if (!metadata.ok()) return metadata;
  absl::Status workspace = bundle.workspace.Release();
  if (!workspace.ok()) return workspace;
  absl::Status input = bundle.input.Release();
  if (!input.ok()) return input;
  return absl::OkStatus();
}

absl::Status CleanupFailedSubmit(CudaTestSubmission::Bundle& bundle, const absl::Status& failure) {
  if (bundle.queued) {
    const CudaApi& api = CudaApi::Production();
    absl::Status synchronized =
        CudaErrorStatus(api.device_synchronize(), "test-pipeline-failure-synchronize",
                        bundle.context->info().ordinal, &bundle.context->health());
    if (!synchronized.ok()) return failure;
  }
  if (bundle.completion.active()) {
    absl::StatusOr<FencePoll> poll = bundle.completion.Poll();
    if (poll.ok() && poll->state != FenceState::kPending) {
      static_cast<void>(bundle.completion.Acknowledge());
    }
  }
  if (bundle.compute_done.recorded()) {
    static_cast<void>(bundle.compute_done.Release());
  }
  static_cast<void>(bundle.metadata.Release());
  static_cast<void>(bundle.workspace.Release());
  if (bundle.output.active()) static_cast<void>(bundle.output.Release());
  if (bundle.input.active()) static_cast<void>(bundle.input.Release());
  return failure;
}

}  // namespace

CudaTestSubmission::CudaTestSubmission(std::shared_ptr<State> state,
                                       std::unique_ptr<Bundle> bundle) noexcept
    : state_(std::move(state)), bundle_(std::move(bundle)) {}

CudaTestSubmission::~CudaTestSubmission() noexcept { Defer(); }

CudaTestSubmission::CudaTestSubmission(CudaTestSubmission&& other) noexcept = default;

CudaTestSubmission& CudaTestSubmission::operator=(CudaTestSubmission&& other) noexcept {
  if (this != &other) {
    Defer();
    state_ = std::move(other.state_);
    bundle_ = std::move(other.bundle_);
  }
  return *this;
}

bool CudaTestSubmission::active() const noexcept { return bundle_ != nullptr; }

absl::StatusOr<FencePoll> CudaTestSubmission::Poll() {
  if (bundle_ == nullptr) {
    return absl::FailedPreconditionError("cuda.test_submission.poll: submission is inactive");
  }
  return bundle_->completion.Poll();
}

absl::StatusOr<BufferLease> CudaTestSubmission::FinishCompleted() {
  if (bundle_ == nullptr) {
    return absl::FailedPreconditionError("cuda.test_submission.finish: submission is inactive");
  }
  absl::StatusOr<FencePoll> poll = bundle_->completion.Poll();
  if (!poll.ok()) return poll.status();
  if (poll->state == FenceState::kPending) return PendingSubmission("finish");
  if (poll->state == FenceState::kFailed) return poll->completion_status;
  absl::Status released = ReleaseBundle(*bundle_, true);
  if (!released.ok()) return released;
  BufferLease output = std::move(bundle_->output);
  bundle_.reset();
  state_.reset();
  return output;
}

void CudaTestSubmission::Defer() noexcept {
  if (bundle_ == nullptr) return;
  if (state_ == nullptr) {
    // Construction guarantees a state for active submissions. Retaining a
    // suspect bundle is safer than releasing resources before GPU completion.
    static_cast<void>(bundle_.release());
    return;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  try {
    state_->deferred.push_back(std::move(bundle_));
  } catch (const std::bad_alloc&) {
    // The deferred vector reserves its full capacity in pipeline construction;
    // this is a final safety fallback if an allocator violates that guarantee.
    static_cast<void>(bundle_.release());
  }
  state_.reset();
}

CudaTestPipeline::CudaTestPipeline(CudaDeviceContext& context) {
  state_ = std::make_shared<CudaTestSubmission::State>(context);
  state_->deferred.reserve(context.info().multiprocessor_count > 0 ? 64U : 1U);
}

CudaTestPipeline::~CudaTestPipeline() noexcept {
  if (state_ != nullptr) static_cast<void>(Close());
}

absl::StatusOr<CudaTestSubmission> CudaTestPipeline::Submit(
    std::span<const std::byte> input, const test::StridedCopyParams& parameters) {
  if (state_ == nullptr || state_->closed) {
    return absl::FailedPreconditionError("cuda.test_pipeline.submit: pipeline is closed");
  }
  CudaDeviceContext& context = *state_->context;
  absl::Status accepting = context.health().CheckAcceptingWork();
  if (!accepting.ok()) return accepting;
  if (input.empty() || parameters.element_count == 0) {
    return absl::InvalidArgumentError("cuda.test_pipeline.input: non-empty test input is required");
  }
  if (input.size() > context.staging_pool().geometry().slot_bytes.value()) {
    return absl::OutOfRangeError("cuda.test_pipeline.input: input exceeds staging slot");
  }

  std::unique_ptr<CudaTestSubmission::Bundle> bundle;
  try {
    bundle = std::make_unique<CudaTestSubmission::Bundle>();
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("cuda.test_pipeline: resource bundle allocation failed");
  }
  bundle->context = &context;

  absl::StatusOr<BufferLease> input_lease = context.staging_pool().Acquire();
  if (!input_lease.ok()) return input_lease.status();
  bundle->input = std::move(*input_lease);
  absl::StatusOr<BufferLease> output_lease = context.staging_pool().Acquire();
  if (!output_lease.ok()) return CleanupFailedSubmit(*bundle, output_lease.status());
  bundle->output = std::move(*output_lease);
  const ByteCount bytes(static_cast<uint64_t>(input.size()));
  absl::StatusOr<CudaMetadataLease> metadata = context.metadata_ring().Acquire();
  if (!metadata.ok()) return CleanupFailedSubmit(*bundle, metadata.status());
  bundle->metadata = std::move(*metadata);
  absl::StatusOr<WorkspaceLease> workspace = context.workspace_pool().Acquire();
  if (!workspace.ok()) return CleanupFailedSubmit(*bundle, workspace.status());
  bundle->workspace = std::move(*workspace);
  absl::StatusOr<MutableBufferView> device_input =
      bundle->workspace.Allocate(bytes, ByteCount(256), WorkspaceTag::kTest);
  if (!device_input.ok()) return CleanupFailedSubmit(*bundle, device_input.status());
  absl::StatusOr<MutableBufferView> device_output =
      bundle->workspace.Allocate(bytes, ByteCount(256), WorkspaceTag::kTest);
  if (!device_output.ok()) return CleanupFailedSubmit(*bundle, device_output.status());
  absl::StatusOr<CudaEventLease> compute_done = context.event_pool().Acquire();
  if (!compute_done.ok()) return CleanupFailedSubmit(*bundle, compute_done.status());
  bundle->compute_done = std::move(*compute_done);
  absl::StatusOr<CudaEventLease> final_event = context.event_pool().Acquire();
  if (!final_event.ok()) return CleanupFailedSubmit(*bundle, final_event.status());

  MutableBufferView input_view = bundle->input.mutable_view().value();
  MutableBufferView output_view = bundle->output.mutable_view().value();
  std::span<std::byte> input_bytes = input_view.HostBytes().value();
  std::copy(input.begin(), input.end(), input_bytes.begin());
  std::fill(output_view.HostBytes()->begin(), output_view.HostBytes()->end(), std::byte{0});
  MutableBufferView metadata_host = bundle->metadata.host_view().value();
  std::memset(metadata_host.HostBytes()->data(), 0, metadata_host.HostBytes()->size());
  std::memcpy(metadata_host.HostBytes()->data(), &parameters,
              std::min(metadata_host.HostBytes()->size(), sizeof(parameters)));
  absl::StatusOr<MutableBufferView> scratch =
      bundle->workspace.Allocate(ByteCount(256), ByteCount(64), WorkspaceTag::kTest);
  if (!scratch.ok()) return CleanupFailedSubmit(*bundle, scratch.status());

  bundle->queued = true;
  absl::Status copy =
      CopyAsync({input_view.AsConst(), *device_input, bytes}, context.transfer_stream(),
                CudaApi::Production(), &context.health());
  if (!copy.ok()) return CleanupFailedSubmit(*bundle, copy);
  absl::StatusOr<BufferView> device_metadata =
      bundle->metadata.SealAndUpload(context.transfer_stream(), context.compute_stream());
  if (!device_metadata.ok()) {
    return CleanupFailedSubmit(*bundle, device_metadata.status());
  }
  test::LaunchStridedCopy(BufferAccess::Address(device_input->AsConst()),
                          BufferAccess::Address(*device_output), parameters,
                          context.compute_stream().handle());
  const cudaError_t launch = CudaApi::Production().peek_at_last_error();
  if (launch != cudaSuccess) {
    return CleanupFailedSubmit(*bundle, CudaErrorStatus(launch, "test-pipeline-launch",
                                                        context.info().ordinal, &context.health()));
  }
  absl::Status recorded = bundle->compute_done.Record(context.compute_stream());
  if (!recorded.ok()) return CleanupFailedSubmit(*bundle, recorded);
  absl::Status waited = bundle->compute_done.WaitOn(context.transfer_stream());
  if (!waited.ok()) return CleanupFailedSubmit(*bundle, waited);
  copy = CopyAsync({device_output->AsConst(), output_view, bytes}, context.transfer_stream(),
                   CudaApi::Production(), &context.health());
  if (!copy.ok()) return CleanupFailedSubmit(*bundle, copy);
  recorded = final_event->Record(context.transfer_stream());
  if (!recorded.ok()) return CleanupFailedSubmit(*bundle, recorded);
  absl::StatusOr<CompletionFence> completion = final_event->IntoFence();
  if (!completion.ok()) return CleanupFailedSubmit(*bundle, completion.status());
  bundle->completion = std::move(*completion);
  return CudaTestSubmission(state_, std::move(bundle));
}

absl::Status CudaTestPipeline::ReclaimDeferred() {
  if (state_ == nullptr) return absl::OkStatus();
  std::lock_guard<std::mutex> lock(state_->mutex);
  size_t destination = 0;
  for (size_t index = 0; index < state_->deferred.size(); ++index) {
    std::unique_ptr<CudaTestSubmission::Bundle>& bundle = state_->deferred[index];
    absl::StatusOr<FencePoll> poll = bundle->completion.Poll();
    if (!poll.ok()) return poll.status();
    if (poll->state == FenceState::kPending) {
      state_->deferred[destination++] = std::move(bundle);
      continue;
    }
    absl::Status released = ReleaseBundle(*bundle, true);
    if (!released.ok()) return released;
    absl::Status output = bundle->output.Release();
    if (!output.ok()) return output;
  }
  state_->deferred.resize(destination);
  return absl::OkStatus();
}

uint32_t CudaTestPipeline::deferred_count() const noexcept {
  if (state_ == nullptr) return 0;
  std::lock_guard<std::mutex> lock(state_->mutex);
  return static_cast<uint32_t>(state_->deferred.size());
}

absl::Status CudaTestPipeline::Close() {
  if (state_ == nullptr || state_->closed) return absl::OkStatus();
  absl::Status reclaimed = ReclaimDeferred();
  if (!reclaimed.ok()) return reclaimed;
  if (deferred_count() != 0) return PendingSubmission("close");
  state_->closed = true;
  return absl::OkStatus();
}

}  // namespace inferx::cuda::testing
