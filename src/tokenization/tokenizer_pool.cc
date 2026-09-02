// Bounded tokenizer pool.

#include "inferx/tokenization/tokenizer_pool.h"

#include <chrono>
#include <mutex>
#include <thread>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace inferx::tokenization {
namespace {

constexpr char kReasonUrl[] = "type.inferx.ai/reason/tokenization";

constexpr std::string_view TokenizationReasonName(TokenizationReason reason) {
  switch (reason) {
    case TokenizationReason::kInputLimit:
      return "kTokenizationInputLimit";
    case TokenizationReason::kQueueFull:
      return "kTokenizationQueueFull";
    case TokenizationReason::kByteBudget:
      return "kTokenizationByteBudget";
    case TokenizationReason::kDeadline:
      return "kTokenizationDeadline";
    case TokenizationReason::kCancelled:
      return "kTokenizationCancelled";
    case TokenizationReason::kPoolClosed:
      return "kTokenizationPoolClosed";
    case TokenizationReason::kInvalidPrompt:
      return "kTokenizationInvalidPrompt";
    case TokenizationReason::kEncodeFailed:
      return "kTokenizationEncodeFailed";
  }
  return "kUnknown";
}

absl::Status Rejected(TokenizationReason reason, absl::StatusCode code, std::string_view message) {
  absl::Status status(code, message);
  status.SetPayload(kReasonUrl, absl::Cord(TokenizationReasonName(reason)));
  return status;
}

}  // namespace

TokenizerPool::TokenizerPool(std::shared_ptr<const Tokenizer> tokenizer, TokenizationLimits limits)
    : tokenizer_(std::move(tokenizer)), limits_(limits) {}

TokenizerPool::~TokenizerPool() { Shutdown(); }

absl::StatusOr<std::unique_ptr<TokenizerPool>> TokenizerPool::Create(
    std::shared_ptr<const Tokenizer> tokenizer, TokenizationLimits limits) {
  if (tokenizer == nullptr) {
    return absl::InvalidArgumentError("tokenizer pool needs a tokenizer");
  }
  if (limits.worker_count < 1 || limits.worker_count > 32) {
    return absl::InvalidArgumentError("tokenization worker_count must be in 1..32");
  }
  if (limits.max_queued_jobs.value() == 0) {
    return absl::InvalidArgumentError("tokenization max_queued_jobs must be positive");
  }
  auto pool = std::unique_ptr<TokenizerPool>(new TokenizerPool(std::move(tokenizer), limits));
  pool->jobs_ = std::make_unique<BoundedChannel<TokenizationJob>>(limits.max_queued_jobs);
  // Completions never block workers: capacity covers queued jobs plus one
  // in flight per worker.
  pool->completions_ = std::make_unique<BoundedChannel<TokenizationCompletion>>(
      QueueCapacity(limits.max_queued_jobs.value() + limits.worker_count));
  for (uint32_t i = 0; i < limits.worker_count; ++i) {
    pool->workers_.emplace_back(
        [pool_ptr = pool.get()](std::stop_token stop) { pool_ptr->WorkerLoop(stop); });
  }
  return pool;
}

void TokenizerPool::WorkerLoop(std::stop_token stop) {
  ++active_workers_;
  for (;;) {
    TokenizationJob job;
    const ChannelResult popped = jobs_->Pop(&job, Deadline::max(), stop);
    if (popped != ChannelResult::kSuccess) break;

    TokenizationCompletion completion;
    completion.request_id = job.request_id;
    completion.epoch = job.epoch;
    completion.input_bytes = static_cast<uint32_t>(job.utf8.size());

    const auto started = std::chrono::steady_clock::now();
    auto encoded = tokenizer_->Encode(job.utf8, job.options);
    completion.encode_ns =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - started)
                                  .count());
    if (encoded.ok()) {
      completion.status = absl::OkStatus();
      completion.output_tokens = static_cast<uint32_t>(encoded->size());
      completion.tokens = std::move(*encoded);
    } else {
      completion.status = encoded.status();
    }

    queued_bytes_.fetch_sub(job.byte_cost);

    // Exactly one completion per accepted job: a completion-slot shortage is
    // a pool sizing bug, so a full channel blocks the worker briefly rather
    // than dropping the completion.
    while (completions_->TryPush(std::move(completion)) == ChannelResult::kFull) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  --active_workers_;
}

absl::Status TokenizerPool::Submit(TokenizationJob job, MonotonicTime now) {
  (void)now;
  {
    std::lock_guard<std::mutex> lock(shutdown_mutex_);
    if (closed_) {
      return Rejected(TokenizationReason::kPoolClosed, absl::StatusCode::kFailedPrecondition,
                      "the tokenization pool is closed");
    }
  }
  if (job.utf8.size() > limits_.max_input_bytes) {
    return Rejected(TokenizationReason::kInputLimit, absl::StatusCode::kResourceExhausted,
                    absl::StrCat("prompt is ", job.utf8.size(), " bytes; the per-request limit is ",
                                 limits_.max_input_bytes));
  }
  job.byte_cost = job.utf8.size();

  // Reserve queued bytes first; roll back if the queue rejects the job.
  uint64_t queued = queued_bytes_.load(std::memory_order_relaxed);
  for (;;) {
    if (queued + job.byte_cost > limits_.max_queued_bytes) {
      return Rejected(TokenizationReason::kByteBudget, absl::StatusCode::kResourceExhausted,
                      "the tokenization queued-byte budget is full");
    }
    if (queued_bytes_.compare_exchange_weak(queued, queued + job.byte_cost,
                                            std::memory_order_relaxed)) {
      break;
    }
  }

  const uint64_t reserved = job.byte_cost;
  const ChannelResult pushed = jobs_->TryPush(std::move(job));
  if (pushed != ChannelResult::kSuccess) {
    queued_bytes_.fetch_sub(reserved);
    if (pushed == ChannelResult::kFull) {
      return Rejected(TokenizationReason::kQueueFull, absl::StatusCode::kResourceExhausted,
                      "the tokenization queue is full");
    }
    return Rejected(TokenizationReason::kPoolClosed, absl::StatusCode::kFailedPrecondition,
                    "the tokenization pool is closed");
  }
  return absl::OkStatus();
}

absl::StatusOr<TokenizationCompletion> TokenizerPool::NextCompletion(Deadline deadline,
                                                                     std::stop_token stop) {
  TokenizationCompletion completion;
  const ChannelResult popped = completions_->Pop(&completion, deadline, stop);
  switch (popped) {
    case ChannelResult::kSuccess:
      return completion;
    case ChannelResult::kTimeout:
      return absl::DeadlineExceededError("no tokenization completion in time");
    case ChannelResult::kStopped:
      return absl::CancelledError("tokenization completion wait stopped");
    case ChannelResult::kClosed:
      return absl::FailedPreconditionError(
          "the tokenization completion channel is closed and drained");
    case ChannelResult::kFull:
    case ChannelResult::kEmpty:
      break;
  }
  return absl::InternalError("unexpected tokenization channel result");
}

void TokenizerPool::Shutdown() {
  {
    std::lock_guard<std::mutex> lock(shutdown_mutex_);
    closed_ = true;
  }
  jobs_->Close();
  // Drain remaining jobs with one cancelled completion each so every
  // accepted job observes exactly one outcome, then let workers exit.
  for (;;) {
    TokenizationJob job;
    if (jobs_->TryPop(&job) != ChannelResult::kSuccess) break;
    TokenizationCompletion cancelled;
    cancelled.request_id = job.request_id;
    cancelled.epoch = job.epoch;
    cancelled.status = Rejected(TokenizationReason::kCancelled, absl::StatusCode::kCancelled,
                                "the tokenization pool shut down before the job ran");
    queued_bytes_.fetch_sub(job.byte_cost);
    completions_->TryPush(std::move(cancelled));
  }
  for (std::jthread& worker : workers_) {
    worker.request_stop();
  }
  for (std::jthread& worker : workers_) {
    if (worker.joinable()) worker.join();
  }
  workers_.clear();
  completions_->Close();
}

}  // namespace inferx::tokenization
