// Preallocated generation-checked CUDA events and CompletionFence domain.
#ifndef INFERX_PLATFORM_CUDA_CUDA_EVENT_POOL_H_
#define INFERX_PLATFORM_CUDA_CUDA_EVENT_POOL_H_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/platform/cuda/cuda_api.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/cuda_stream.h"
#include "inferx/runtime/completion_fence.h"

namespace inferx::cuda {

class CudaEventPool;

class CudaEventLease {
 public:
  CudaEventLease() noexcept = default;
  ~CudaEventLease() noexcept;
  CudaEventLease(CudaEventLease&& other) noexcept;
  CudaEventLease& operator=(CudaEventLease&& other) noexcept;
  CudaEventLease(const CudaEventLease&) = delete;
  CudaEventLease& operator=(const CudaEventLease&) = delete;

  [[nodiscard]] FenceToken token() const noexcept { return token_; }
  [[nodiscard]] bool recorded() const noexcept { return recorded_; }
  [[nodiscard]] absl::Status Record(CudaStream& stream);
  [[nodiscard]] absl::Status WaitOn(CudaStream& stream);
  [[nodiscard]] absl::StatusOr<FencePoll> Poll();
  [[nodiscard]] absl::Status Release();
  [[nodiscard]] absl::StatusOr<CompletionFence> IntoFence();

 private:
  friend class CudaEventPool;
  CudaEventLease(CudaEventPool* pool, FenceToken token) noexcept;
  CudaEventPool* pool_ = nullptr;
  FenceToken token_{Device::Cuda(DeviceId(0)), FenceSlotId(0), FenceGeneration(0)};
  bool recorded_ = false;
};

class CudaEventPool final : public FenceDomain {
 public:
  [[nodiscard]] static absl::StatusOr<std::unique_ptr<CudaEventPool>> Create(
      DeviceId device, uint32_t slots, const CudaApi& api = CudaApi::Production(),
      CudaHealth* health = nullptr);
  ~CudaEventPool() noexcept override;
  CudaEventPool(const CudaEventPool&) = delete;
  CudaEventPool& operator=(const CudaEventPool&) = delete;

  [[nodiscard]] absl::StatusOr<CudaEventLease> Acquire();
  [[nodiscard]] absl::Status ReclaimAbandoned();
  [[nodiscard]] absl::Status Close();
  [[nodiscard]] uint32_t available_slots() const noexcept;

  [[nodiscard]] absl::StatusOr<FencePoll> Poll(FenceToken token) override;
  [[nodiscard]] absl::Status WaitUntil(FenceToken token, Deadline deadline,
                                       FenceWaitReason reason) override;
  [[nodiscard]] absl::Status Acknowledge(FenceToken token) override;
  void Abandon(FenceToken token) noexcept override;

 private:
  friend class CudaEventLease;
  struct Impl;
  explicit CudaEventPool(std::unique_ptr<Impl> impl) noexcept;
  [[nodiscard]] absl::Status Record(FenceToken token, CudaStream& stream);
  [[nodiscard]] absl::Status WaitOn(FenceToken token, CudaStream& stream);
  [[nodiscard]] absl::Status ReleaseLease(FenceToken token);
  [[nodiscard]] absl::Status ReleaseUnrecorded(FenceToken token);
  [[nodiscard]] absl::StatusOr<cudaEvent_t> Resolve(FenceToken token, bool require_recorded);
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::cuda

#endif  // INFERX_PLATFORM_CUDA_CUDA_EVENT_POOL_H_
