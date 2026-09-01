// Hardware-neutral move-only completion observation and acknowledgement.
#ifndef INFERX_RUNTIME_COMPLETION_FENCE_H_
#define INFERX_RUNTIME_COMPLETION_FENCE_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/tensor/device.h"

namespace inferx {

enum class FenceState : uint8_t { kPending, kComplete, kFailed };
enum class FenceWaitReason : uint8_t { kTest, kStartup, kDiagnostic, kShutdown };

struct FenceToken {
  Device device;
  FenceSlotId slot;
  FenceGeneration generation;

  friend constexpr bool operator==(const FenceToken&, const FenceToken&) = default;
};

struct FencePoll {
  FenceState state;
  absl::Status completion_status;
};

class FenceDomain {
 public:
  virtual ~FenceDomain() = default;
  [[nodiscard]] virtual absl::StatusOr<FencePoll> Poll(FenceToken token) = 0;
  [[nodiscard]] virtual absl::Status WaitUntil(FenceToken token, Deadline deadline,
                                               FenceWaitReason reason) = 0;
  [[nodiscard]] virtual absl::Status Acknowledge(FenceToken token) = 0;
  virtual void Abandon(FenceToken token) noexcept = 0;
};

class CompletionFence {
 public:
  CompletionFence() noexcept = default;
  CompletionFence(FenceDomain* domain, FenceToken token) noexcept;
  ~CompletionFence() noexcept;
  CompletionFence(CompletionFence&& other) noexcept;
  CompletionFence& operator=(CompletionFence&& other) noexcept;
  CompletionFence(const CompletionFence&) = delete;
  CompletionFence& operator=(const CompletionFence&) = delete;

  [[nodiscard]] bool active() const noexcept { return domain_ != nullptr; }
  [[nodiscard]] FenceToken token() const noexcept { return token_; }
  [[nodiscard]] absl::StatusOr<FencePoll> Poll();
  [[nodiscard]] absl::Status WaitUntil(Deadline deadline, FenceWaitReason reason);
  [[nodiscard]] absl::Status Acknowledge();

 private:
  void Abandon() noexcept;

  FenceDomain* domain_ = nullptr;
  FenceToken token_{Device::Host(), FenceSlotId(0), FenceGeneration(0)};
  bool terminal_observed_ = false;
};

}  // namespace inferx

#endif  // INFERX_RUNTIME_COMPLETION_FENCE_H_
