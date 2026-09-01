#ifndef INFERX_TESTS_FAKES_FAKE_FENCE_H_
#define INFERX_TESTS_FAKES_FAKE_FENCE_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/clock.h"
#include "inferx/runtime/completion_fence.h"

namespace inferx::testing {

class FakeFenceDomain final : public FenceDomain {
 public:
  explicit FakeFenceDomain(uint32_t slot_count,
                           FenceGeneration initial_generation = FenceGeneration(0));

  [[nodiscard]] absl::StatusOr<CompletionFence> Acquire(Device device = Device::Host());
  [[nodiscard]] absl::Status Complete(FenceToken token);
  [[nodiscard]] absl::Status Fail(FenceToken token, absl::Status status);
  [[nodiscard]] uint32_t abandoned_count() const noexcept;
  [[nodiscard]] absl::Status ReclaimAbandoned();

  [[nodiscard]] absl::StatusOr<FencePoll> Poll(FenceToken token) override;
  [[nodiscard]] absl::Status WaitUntil(FenceToken token, Deadline deadline,
                                       FenceWaitReason reason) override;
  [[nodiscard]] absl::Status Acknowledge(FenceToken token) override;
  void Abandon(FenceToken token) noexcept override;

 private:
  struct Slot {
    FenceGeneration generation = FenceGeneration(0);
    FenceState state = FenceState::kPending;
    absl::Status status;
    bool active = false;
    bool abandoned = false;
    Device device = Device::Host();
  };

  [[nodiscard]] absl::StatusOr<Slot*> Resolve(FenceToken token);
  std::vector<Slot> slots_;
};

}  // namespace inferx::testing

#endif  // INFERX_TESTS_FAKES_FAKE_FENCE_H_
