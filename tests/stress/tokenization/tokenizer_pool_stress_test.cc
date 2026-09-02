// Tokenizer pool stress (ADR 0024 TSan gate):
// concurrent submitters and completion consumers plus parallel streaming
// decoders borrowing pool instances. Every accepted job observes exactly one
// completion, byte accounting returns to zero, and the run must be clean
// under ThreadSanitizer.
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/base/id.h"
#include "inferx/tokenization/tokenizer.h"
#include "inferx/tokenization/tokenizer_pool.h"

namespace inferx::tokenization {
namespace {

std::string ReadFile(const char* path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

TEST(TokenizerPoolStress, ExactlyOnceCompletionsAndZeroBudgetAtRest) {
  const std::string tokenizer_json = ReadFile(TOKENIZER_JSON_PATH);
  ASSERT_FALSE(tokenizer_json.empty());

  TokenizerArtifacts artifacts;
  artifacts.tokenizer_json = tokenizer_json;
  TokenizerInstancePolicy policy;
  policy.instance_count = 4;
  auto loaded = Tokenizer::Load(std::move(artifacts), policy);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  auto tokenizer = std::shared_ptr<const Tokenizer>(std::move(*loaded));

  TokenizationLimits limits;
  limits.worker_count = 4;
  limits.max_queued_jobs = QueueCapacity(64);
  limits.max_queued_bytes = 8u << 20;
  auto pool_result = TokenizerPool::Create(tokenizer, limits);
  ASSERT_TRUE(pool_result.ok()) << pool_result.status();
  std::unique_ptr<TokenizerPool> pool = std::move(*pool_result);

  constexpr int kJobs = 400;
  constexpr int kSubmitters = 8;
  std::atomic<int> submitted{0};
  std::atomic<int> rejected{0};
  std::atomic<bool> done_submitting{false};

  std::vector<std::thread> submitters;
  for (int t = 0; t < kSubmitters; ++t) {
    submitters.emplace_back([&, t] {
      for (int i = t; i < kJobs; i += kSubmitters) {
        TokenizationJob job;
        job.request_id = RequestId(static_cast<uint64_t>(i));
        job.epoch = RequestEpoch(0);
        job.utf8 = "stress request " + std::to_string(i) +
                   " with a little multilingual tail: 一段中文, Привет, 🚀";
        EncodeOptions options;
        options.add_special_tokens = true;
        job.options = options;
        auto status = pool->Submit(std::move(job), std::chrono::steady_clock::now());
        if (status.ok()) {
          submitted.fetch_add(1);
        } else {
          rejected.fetch_add(1);
        }
      }
    });
  }

  // Concurrent streaming decoders borrowing the same instance set.
  std::vector<std::thread> decoders;
  for (int t = 0; t < 2; ++t) {
    decoders.emplace_back([&] {
      while (!done_submitting.load()) {
        auto encoded = tokenizer->Encode("stream stress Привет 🚀", EncodeOptions{});
        if (!encoded.ok()) continue;
        auto decoder = tokenizer->NewIncrementalDecoder(DecodeOptions{});
        if (!decoder.ok()) continue;
        for (TokenId id : *encoded) {
          auto chunk = (*decoder)->Push(id);
          if (!chunk.ok()) break;
        }
        auto finish = (*decoder)->Finish();
        if (!finish.ok()) continue;
      }
    });
  }

  // Submitters are joined first so the accepted count is final; only
  // accepted jobs are owed a completion (rejected jobs never enqueue).
  for (std::thread& t : submitters) t.join();
  const int accepted = submitted.load();
  ASSERT_EQ(accepted + rejected.load(), kJobs);

  std::unordered_set<uint64_t> seen;
  std::mutex seen_mutex;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (static_cast<int>(seen.size()) < accepted) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "timed out with " << seen.size() << "/" << accepted << " completions";
    auto completion = pool->NextCompletion(
        std::chrono::steady_clock::now() + std::chrono::milliseconds(100), std::stop_token{});
    if (!completion.ok()) continue;
    std::lock_guard<std::mutex> lock(seen_mutex);
    const uint64_t id = completion->request_id.value();
    ASSERT_TRUE(seen.insert(id).second) << "duplicate completion for request " << id;
    ASSERT_TRUE(completion->status.ok()) << "unexpected pool error: " << completion->status;
  }
  done_submitting = true;

  for (std::thread& t : decoders) t.join();

  pool->Shutdown();
  // Drain any completions still queued after shutdown.
  for (;;) {
    auto completion = pool->NextCompletion(
        std::chrono::steady_clock::now() + std::chrono::milliseconds(50), std::stop_token{});
    if (!completion.ok()) break;
    std::lock_guard<std::mutex> lock(seen_mutex);
    const uint64_t id = completion->request_id.value();
    ASSERT_TRUE(seen.insert(id).second) << "duplicate completion for request " << id;
  }

  EXPECT_EQ(static_cast<int>(seen.size()), accepted);
  EXPECT_EQ(pool->queued_bytes(), 0u);
  EXPECT_EQ(pool->active_workers(), 0u);
}

}  // namespace
}  // namespace inferx::tokenization
