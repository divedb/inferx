// Bounded tokenizer pool.
//
// Owns exactly `worker_count` exclusive engine instances before readiness.
// Jobs arrive on a bounded channel with byte accounting; every accepted job
// gets exactly one completion even on cancellation or shutdown. Workers
// never call back into the engine coordinator; they place value-only
// completions on the completion channel.

#ifndef INFERX_TOKENIZATION_TOKENIZER_POOL_H_
#define INFERX_TOKENIZATION_TOKENIZER_POOL_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/base/bounded_channel.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/tokenization/tokenizer.h"
#include "inferx/tokenization/tokenizer_options.h"

namespace inferx::tokenization {

// Reason codes for pool rejections; part of the stable status payload.
enum class TokenizationReason : uint8_t {
  kInputLimit,
  kQueueFull,
  kByteBudget,
  kDeadline,
  kCancelled,
  kPoolClosed,
  kInvalidPrompt,
  kEncodeFailed,
};

struct TokenizationLimits {
  // 1..min(32, hardware_concurrency or 1); fixed at startup.
  uint32_t worker_count = 1;
  QueueCapacity max_queued_jobs{256};
  // Exact owned prompt/template bytes awaiting work.
  uint64_t max_queued_bytes = 64u << 20;
  // Per request before template rendering.
  uint64_t max_input_bytes = 4u << 20;
  // Checked while rendering, before encode.
  uint64_t max_rendered_bytes = 8u << 20;
};

struct TokenizationJob {
  // Channel slots must be default-constructible; the zero IDs are transient
  // storage state and never an emitted value.
  RequestId request_id = RequestId(0);
  RequestEpoch epoch = RequestEpoch(0);
  std::string utf8;  // owned prompt bytes (already rendered if chat)
  EncodeOptions options;
  uint64_t byte_cost = 0;  // reserved queued bytes; equals utf8.size().
};

struct TokenizationCompletion {
  // See TokenizationJob: zero IDs are transient channel-slot state only.
  RequestId request_id = RequestId(0);
  RequestEpoch epoch = RequestEpoch(0);
  absl::Status status;          // carries TokenizationReason payloads
  std::vector<TokenId> tokens;  // empty unless status is ok
  // Observed timings/counts as values; never prompt text.
  uint64_t queue_wait_ns = 0;
  uint64_t encode_ns = 0;
  uint32_t input_bytes = 0;
  uint32_t output_tokens = 0;
};

class TokenizerPool {
 public:
  // Constructs the pool over `tokenizer`'s engine instances. The facade must
  // have been loaded with at least `limits.worker_count` instances.
  static absl::StatusOr<std::unique_ptr<TokenizerPool>> Create(
      std::shared_ptr<const Tokenizer> tokenizer, TokenizationLimits limits);

  ~TokenizerPool();

  // Reserves queued bytes, then enqueues. Exactly one completion for every
  // accepted job; rejected jobs return a status and never enqueue.
  absl::Status Submit(TokenizationJob job, MonotonicTime now);

  // Blocks for the next completion from the worker side.
  absl::StatusOr<TokenizationCompletion> NextCompletion(Deadline deadline, std::stop_token stop);

  const TokenizationLimits& limits() const { return limits_; }
  uint32_t active_workers() const { return active_workers_; }
  uint64_t queued_bytes() const { return queued_bytes_.load(std::memory_order_relaxed); }

  // Fixed shutdown order: stop admission, request stop, drain
  // queued jobs with one cancelled completion each, join workers.
  void Shutdown();

 private:
  TokenizerPool(std::shared_ptr<const Tokenizer> tokenizer, TokenizationLimits limits);
  void WorkerLoop(std::stop_token stop);

  std::shared_ptr<const Tokenizer> tokenizer_;
  TokenizationLimits limits_;
  std::atomic<uint64_t> queued_bytes_{0};
  std::atomic<uint32_t> active_workers_{0};
  std::vector<std::jthread> workers_;
  std::unique_ptr<BoundedChannel<TokenizationJob>> jobs_;
  std::unique_ptr<BoundedChannel<TokenizationCompletion>> completions_;
  std::mutex shutdown_mutex_;
  bool closed_ = false;
};

}  // namespace inferx::tokenization

#endif  // INFERX_TOKENIZATION_TOKENIZER_POOL_H_
