#include "inferx/runtime/contiguous_kv_cache.h"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/base/id.h"
#include "inferx/engine/execution_ticket.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::runtime {
namespace {

absl::Status CacheError(std::string_view detail) {
  return absl::FailedPreconditionError(absl::StrCat("kv_cache: ", detail));
}

}  // namespace

ContiguousKvCache::AppendTransaction::AppendTransaction(AppendTransaction&& other) noexcept
    : cache_(other.cache_), begin_(other.begin_), end_(other.end_), settled_(other.settled_) {
  other.cache_ = nullptr;
  other.settled_ = true;
}

ContiguousKvCache::AppendTransaction& ContiguousKvCache::AppendTransaction::operator=(
    AppendTransaction&& other) noexcept {
  if (this != &other) {
    if (cache_ != nullptr && !settled_) {
      cache_->SettleAppend(false, 0);
    }
    cache_ = other.cache_;
    begin_ = other.begin_;
    end_ = other.end_;
    settled_ = other.settled_;
    other.cache_ = nullptr;
    other.settled_ = true;
  }
  return *this;
}

ContiguousKvCache::AppendTransaction::~AppendTransaction() {
  if (cache_ != nullptr && !settled_) {
    cache_->SettleAppend(false, 0);
  }
}

void ContiguousKvCache::AppendTransaction::Commit() noexcept {
  if (cache_ != nullptr && !settled_) {
    cache_->SettleAppend(true, end_);
    settled_ = true;
  }
}

void ContiguousKvCache::AppendTransaction::Rollback() noexcept {
  if (cache_ != nullptr && !settled_) {
    cache_->SettleAppend(false, 0);
    settled_ = true;
  }
}

ContiguousKvCache::ContiguousKvCache(uint32_t layers, uint64_t capacity,
                                     std::vector<MutableTensorView> keys,
                                     std::vector<MutableTensorView> values)
    : layers_(layers),
      capacity_(capacity),
      key_views_(std::move(keys)),
      value_views_(std::move(values)) {}

absl::StatusOr<ContiguousKvCache> ContiguousKvCache::Create(
    uint32_t layers, uint64_t capacity, std::span<MutableTensorView> key_views,
    std::span<MutableTensorView> value_views) {
  if (layers == 0 || capacity == 0) {
    return absl::InvalidArgumentError("kv_cache: layers and capacity must be positive");
  }
  if (key_views.size() != layers || value_views.size() != layers) {
    return absl::InvalidArgumentError("kv_cache: one view per layer is required");
  }
  for (uint32_t layer = 0; layer < layers; ++layer) {
    if (key_views[layer].shape().rank() != 4 || key_views[layer].shape().dim(0) != 1 ||
        key_views[layer].shape().dim(1) != capacity ||
        value_views[layer].shape() != key_views[layer].shape()) {
      return absl::InvalidArgumentError(
          "kv_cache: per-layer views must be [1, capacity, kv_heads, head_dim]");
    }
  }
  return ContiguousKvCache(layers, capacity,
                           std::vector<MutableTensorView>(key_views.begin(), key_views.end()),
                           std::vector<MutableTensorView>(value_views.begin(), value_views.end()));
}

absl::Status ContiguousKvCache::Acquire(SequenceId sequence) {
  if (state_ != State::kEmpty) return CacheError("acquire requires the empty state");
  sequence_ = sequence;
  committed_ = 0;
  state_ = State::kActive;
  return absl::OkStatus();
}

absl::StatusOr<ContiguousKvCache::AppendTransaction> ContiguousKvCache::PrepareAppend(
    uint64_t expected_begin, uint64_t count, ExecutionTicketId /*ticket*/) {
  if (state_ != State::kActive) return CacheError("append requires the active state");
  if (expected_begin != committed_) {
    return absl::FailedPreconditionError(absl::StrCat("kv_cache.append: expected begin ",
                                                      expected_begin, " but committed length is ",
                                                      committed_));
  }
  if (count == 0 || expected_begin + count > capacity_) {
    return absl::OutOfRangeError(absl::StrCat("kv_cache.append: [", expected_begin, ",",
                                              expected_begin + count, ") exceeds capacity ",
                                              capacity_));
  }
  state_ = State::kAppendPending;
  return AppendTransaction(*this, expected_begin, expected_begin + count);
}

void ContiguousKvCache::SettleAppend(bool commit, uint64_t end) noexcept {
  if (state_ != State::kAppendPending) return;
  if (commit) committed_ = end;  // no-fail: range validated by PrepareAppend
  state_ = State::kActive;
}

absl::Status ContiguousKvCache::Reset() {
  if (state_ == State::kAppendPending) return CacheError("reset with a pending append");
  if (state_ != State::kActive) return CacheError("reset requires the active state");
  const uint32_t next = generation_.value() + 1;
  if (next == 0) return CacheError("generation overflow");
  generation_ = KvSequenceGeneration(next);
  committed_ = 0;
  // Back to the re-acquirable state: the next sequence Acquires cleanly.
  state_ = State::kEmpty;
  return absl::OkStatus();
}

void ContiguousKvCache::Poison() noexcept { state_ = State::kPoisoned; }

absl::Status ContiguousKvCache::BeginClosing() noexcept {
  if (state_ == State::kAppendPending) return CacheError("close with a pending append");
  state_ = State::kClosing;
  return absl::OkStatus();
}

}  // namespace inferx::runtime
