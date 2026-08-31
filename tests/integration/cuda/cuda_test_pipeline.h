// Test-only asynchronous CUDA pipeline and resource-bundle contract.
#ifndef INFERX_TESTS_INTEGRATION_CUDA_CUDA_TEST_PIPELINE_H_
#define INFERX_TESTS_INTEGRATION_CUDA_CUDA_TEST_PIPELINE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/platform/cuda/cuda_device_context.h"
#include "inferx/runtime/buffer_pool.h"
#include "inferx/runtime/completion_fence.h"
#include "test_kernels.h"

namespace inferx::cuda::testing {

class CudaTestPipeline;

class CudaTestSubmission {
 public:
  struct Bundle;
  struct State;

  CudaTestSubmission() noexcept = default;
  ~CudaTestSubmission() noexcept;
  CudaTestSubmission(CudaTestSubmission&& other) noexcept;
  CudaTestSubmission& operator=(CudaTestSubmission&& other) noexcept;
  CudaTestSubmission(const CudaTestSubmission&) = delete;
  CudaTestSubmission& operator=(const CudaTestSubmission&) = delete;

  [[nodiscard]] bool active() const noexcept;
  [[nodiscard]] absl::StatusOr<FencePoll> Poll();
  // Output is intentionally unavailable until this method observes terminal
  // completion, acknowledges the fence, and releases every earlier resource.
  [[nodiscard]] absl::StatusOr<BufferLease> FinishCompleted();

 private:
  friend class CudaTestPipeline;
  CudaTestSubmission(std::shared_ptr<State> state, std::unique_ptr<Bundle> bundle) noexcept;
  void Defer() noexcept;

  std::shared_ptr<State> state_;
  std::unique_ptr<Bundle> bundle_;
};

class CudaTestPipeline {
 public:
  explicit CudaTestPipeline(CudaDeviceContext& context);
  ~CudaTestPipeline() noexcept;
  CudaTestPipeline(CudaTestPipeline&&) noexcept = default;
  CudaTestPipeline& operator=(CudaTestPipeline&&) noexcept = default;
  CudaTestPipeline(const CudaTestPipeline&) = delete;
  CudaTestPipeline& operator=(const CudaTestPipeline&) = delete;

  [[nodiscard]] absl::StatusOr<CudaTestSubmission> Submit(
      std::span<const std::byte> input, const test::StridedCopyParams& parameters);
  [[nodiscard]] absl::Status ReclaimDeferred();
  [[nodiscard]] uint32_t deferred_count() const noexcept;
  [[nodiscard]] absl::Status Close();

 private:
  std::shared_ptr<CudaTestSubmission::State> state_;
};

}  // namespace inferx::cuda::testing

#endif  // INFERX_TESTS_INTEGRATION_CUDA_CUDA_TEST_PIPELINE_H_
