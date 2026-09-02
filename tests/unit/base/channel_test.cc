// Bounded channel semantics tests (ADR 0013). The TSan lane runs these tests.
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "inferx/base/bounded_channel.h"
#include "inferx/base/clock.h"

namespace {

using inferx::BoundedChannel;
using inferx::ChannelResult;
using inferx::Deadline;
using inferx::MonotonicTime;
using inferx::Nanoseconds;
using inferx::QueueCapacity;
using inferx::SteadyClock;

Deadline FarFuture() { return SteadyClock{}.Now() + Nanoseconds(2'000'000'000); }

Deadline Expired() { return SteadyClock{}.Now() - Nanoseconds(1'000'000); }

TEST(BoundedChannelTest, FifoCapacityOne) {
  BoundedChannel<int> channel(QueueCapacity(1));
  EXPECT_EQ(channel.capacity(), 1u);
  EXPECT_EQ(channel.TryPush(7), ChannelResult::kSuccess);
  EXPECT_EQ(channel.size(), 1u);
  EXPECT_EQ(channel.TryPush(8), ChannelResult::kFull);

  int out = 0;
  EXPECT_EQ(channel.TryPop(&out), ChannelResult::kSuccess);
  EXPECT_EQ(out, 7);
  EXPECT_EQ(channel.TryPop(&out), ChannelResult::kEmpty);
}

TEST(BoundedChannelTest, FifoOrderCapacityN) {
  BoundedChannel<int> channel(QueueCapacity(8));
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQ(channel.TryPush(int{i}), ChannelResult::kSuccess);
  }
  for (int i = 0; i < 8; ++i) {
    int out = -1;
    ASSERT_EQ(channel.TryPop(&out), ChannelResult::kSuccess);
    EXPECT_EQ(out, i);
  }
}

TEST(BoundedChannelTest, CloseRejectsPushAndDrains) {
  BoundedChannel<int> channel(QueueCapacity(4));
  ASSERT_EQ(channel.TryPush(1), ChannelResult::kSuccess);
  channel.Close();
  channel.Close();  // idempotent
  EXPECT_EQ(channel.TryPush(2), ChannelResult::kClosed);

  int out = 0;
  EXPECT_EQ(channel.TryPop(&out), ChannelResult::kSuccess);  // drain first
  EXPECT_EQ(out, 1);
  EXPECT_EQ(channel.TryPop(&out), ChannelResult::kClosed);  // then closed

  EXPECT_EQ(channel.Pop(&out, FarFuture(), std::stop_source{}.get_token()), ChannelResult::kClosed);
}

TEST(BoundedChannelTest, PushWakesOnClose) {
  BoundedChannel<int> channel(QueueCapacity(1));
  ASSERT_EQ(channel.TryPush(0), ChannelResult::kSuccess);  // force blocking
  std::jthread waiter([&] {
    EXPECT_EQ(channel.Push(int{5}, FarFuture(), std::stop_source{}.get_token()),
              ChannelResult::kClosed);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  channel.Close();
}

TEST(BoundedChannelTest, ExpiredDeadlineTimesOutWithoutConsumingValue) {
  BoundedChannel<std::unique_ptr<int>> channel(QueueCapacity(1));
  ASSERT_EQ(channel.TryPush(std::make_unique<int>(1)), ChannelResult::kSuccess);

  std::unique_ptr<int> blocked = std::make_unique<int>(2);
  EXPECT_EQ(channel.Push(std::move(blocked), Expired(), std::stop_source{}.get_token()),
            ChannelResult::kTimeout);
  // Untouched on failure: the caller may retry with the same move-only
  // value; the timeout path provably did not move from it.
  ASSERT_NE(blocked, nullptr);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(*blocked, 2);       // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(channel.size(), 1u);
}

TEST(BoundedChannelTest, PopTimeoutOnEmpty) {
  BoundedChannel<int> channel(QueueCapacity(2));
  int out = 0;
  EXPECT_EQ(channel.Pop(&out, Expired(), std::stop_source{}.get_token()), ChannelResult::kTimeout);
}

TEST(BoundedChannelTest, StopNeverTransfersValue) {
  BoundedChannel<int> channel(QueueCapacity(1));
  std::stop_source stop;
  stop.request_stop();
  int value = 9;
  EXPECT_EQ(channel.Push(int{value}, FarFuture(), stop.get_token()), ChannelResult::kStopped);
  EXPECT_EQ(value, 9);  // untouched

  // Pop with a pre-stopped token: stop wins even though data is queued.
  ASSERT_EQ(channel.TryPush(10), ChannelResult::kSuccess);
  int out = 0;
  EXPECT_EQ(channel.Pop(&out, FarFuture(), stop.get_token()), ChannelResult::kStopped);
  EXPECT_EQ(out, 0);  // not written
}

TEST(BoundedChannelTest, StopWakesBlockedPop) {
  BoundedChannel<int> channel(QueueCapacity(2));
  std::stop_source stop;
  int out = 0;
  std::jthread waiter([&] {
    EXPECT_EQ(channel.Pop(&out, FarFuture(), stop.get_token()), ChannelResult::kStopped);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  stop.request_stop();
}

TEST(BoundedChannelTest, BlockingProducerConsumerHandshake) {
  BoundedChannel<int> channel(QueueCapacity(1));
  std::vector<int> received;
  received.reserve(64);
  std::jthread consumer([&] {
    const Deadline deadline = FarFuture();
    while (true) {
      int out = 0;
      const ChannelResult result = channel.Pop(&out, deadline, std::stop_source{}.get_token());
      if (result == ChannelResult::kClosed) break;
      ASSERT_EQ(result, ChannelResult::kSuccess);
      received.push_back(out);
    }
  });
  const Deadline deadline = FarFuture();
  for (int i = 0; i < 64; ++i) {
    ASSERT_EQ(channel.Push(int{i}, deadline, std::stop_source{}.get_token()),
              ChannelResult::kSuccess);
  }
  channel.Close();
  consumer.join();  // observe results only after the drain completes
  ASSERT_EQ(received.size(), 64u);
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(received[static_cast<size_t>(i)], i);
  }
}

TEST(BoundedChannelTest, MultiProducerMultiConsumerStress) {
  constexpr int kProducers = 4;
  constexpr int kConsumers = 3;
  constexpr int kPerProducer = 512;
  BoundedChannel<int> channel(QueueCapacity(7));

  std::vector<std::vector<int>> consumed(kConsumers);
  for (auto& shard : consumed) {
    shard.reserve(kPerProducer);
  }
  std::vector<std::jthread> consumers;
  consumers.reserve(kConsumers);
  for (int c = 0; c < kConsumers; ++c) {
    consumers.emplace_back([&channel, &consumed, c] {
      const Deadline deadline = FarFuture();
      while (true) {
        int out = 0;
        const ChannelResult result = channel.Pop(&out, deadline, std::stop_source{}.get_token());
        if (result == ChannelResult::kClosed) break;
        ASSERT_EQ(result, ChannelResult::kSuccess);
        consumed[static_cast<size_t>(c)].push_back(out);
      }
    });
  }

  std::vector<std::jthread> producers;
  producers.reserve(kProducers);
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&channel, p] {
      const Deadline deadline = FarFuture();
      const int base = p * kPerProducer;
      for (int i = 0; i < kPerProducer; ++i) {
        ASSERT_EQ(channel.Push(base + i, deadline, std::stop_source{}.get_token()),
                  ChannelResult::kSuccess);
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  channel.Close();
  for (auto& consumer : consumers) {
    consumer.join();
  }

  // Every value exactly once; cross-consumer interleaving is allowed, so
  // order is verified by sorting (the ring's FIFO holds per-slot order).
  std::vector<int> expected;
  for (int p = 0; p < kProducers; ++p) {
    for (int i = 0; i < kPerProducer; ++i) {
      expected.push_back(p * kPerProducer + i);
    }
  }
  std::vector<int> seen;
  seen.reserve(static_cast<size_t>(kProducers) * kPerProducer);
  for (const auto& shard : consumed) {
    seen.insert(seen.end(), shard.begin(), shard.end());
  }
  ASSERT_EQ(seen.size(), expected.size());
  std::sort(seen.begin(), seen.end());
  EXPECT_EQ(seen, expected);
}

}  // namespace
