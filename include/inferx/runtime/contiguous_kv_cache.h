#pragma once

#include <cstdint>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/engine/execution_ticket.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::runtime {

struct KvSequenceGenerationTag {};
using KvSequenceGeneration = StrongId<KvSequenceGenerationTag, uint32_t>;

// Correctness-oriented contiguous KV cache for one sequence (m5.md section
// 11). The cache owns checked views and state only: the ModelInstance owns
// the backing storage. Batch one; M6 replaces the physical layout behind
// this interface.
class ContiguousKvCache {
 public:
  enum class State : uint8_t { kEmpty, kActive, kAppendPending, kPoisoned, kClosing };

  // Move-only append transaction: launch may write the destinations, but the
  // committed length advances only through Commit after the matching
  // execution completion succeeds.
  class AppendTransaction {
   public:
    AppendTransaction(ContiguousKvCache& cache, uint64_t begin, uint64_t end) noexcept
        : cache_(&cache), begin_(begin), end_(end) {}
    AppendTransaction(AppendTransaction&& other) noexcept;
    AppendTransaction& operator=(AppendTransaction&&) noexcept;
    AppendTransaction(const AppendTransaction&) = delete;
    AppendTransaction& operator=(const AppendTransaction&) = delete;
    ~AppendTransaction();

    [[nodiscard]] uint64_t begin() const noexcept { return begin_; }
    [[nodiscard]] uint64_t end() const noexcept { return end_; }
    // No-fail length advance; called only after the covering fence succeeded.
    void Commit() noexcept;
    // Synchronous pre-launch failure: length unchanged.
    void Rollback() noexcept;

   private:
    ContiguousKvCache* cache_;
    uint64_t begin_;
    uint64_t end_;
    bool settled_ = false;
  };

  // `key_views`/`value_views` carry one [1, capacity, kv_heads, head_dim]
  // view per layer over model-owned backing storage.
  static absl::StatusOr<ContiguousKvCache> Create(uint32_t layers, uint64_t capacity,
                                                  std::span<MutableTensorView> key_views,
                                                  std::span<MutableTensorView> value_views);

  // empty -> active; assigns sequence/generation with committed length zero.
  [[nodiscard]] absl::Status Acquire(SequenceId sequence);

  // Validated append reservation at the committed boundary.
  [[nodiscard]] absl::StatusOr<AppendTransaction> PrepareAppend(uint64_t expected_begin,
                                                                uint64_t count,
                                                                ExecutionTicketId ticket);

  // No pending transaction; increments generation, drops logical visibility
  // to zero. Bytes are not erased: visibility is the committed length.
  [[nodiscard]] absl::Status Reset();

  void Poison() noexcept;
  [[nodiscard]] absl::Status BeginClosing() noexcept;

  [[nodiscard]] State state() const noexcept { return state_; }
  [[nodiscard]] uint64_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] uint64_t committed_length() const noexcept { return committed_; }
  [[nodiscard]] SequenceId sequence() const noexcept { return sequence_; }
  [[nodiscard]] KvSequenceGeneration generation() const noexcept { return generation_; }
  // Full-capacity per-layer views; attention writes only its append range.
  [[nodiscard]] std::span<MutableTensorView> key_views() noexcept { return key_views_; }
  [[nodiscard]] std::span<MutableTensorView> value_views() noexcept { return value_views_; }

 private:
  ContiguousKvCache(uint32_t layers, uint64_t capacity, std::vector<MutableTensorView> keys,
                    std::vector<MutableTensorView> values);

  friend class AppendTransaction;
  void SettleAppend(bool commit, uint64_t end) noexcept;

  State state_ = State::kEmpty;
  uint32_t layers_ = 0;
  uint64_t capacity_ = 0;
  uint64_t committed_ = 0;
  SequenceId sequence_{0};
  KvSequenceGeneration generation_{0};
  std::vector<MutableTensorView> key_views_;
  std::vector<MutableTensorView> value_views_;
};

}  // namespace inferx::runtime
