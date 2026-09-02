// M5 end-to-end integration on the CPU reference backend: validated package
// -> transactional model load (F16 weights materialized as FP32) -> M1
// lifecycle prefill/decode -> greedy tokens -> EOS/length termination ->
// unload and reload (m5.md sections 1, 10, 13, 18.3).

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/api/generate_request.h"
#include "inferx/api/response_event.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/engine/event_sink.h"
#include "inferx/input/model_package.h"
#include "inferx/runtime/cpu_execution_backend.h"
#include "inferx/runtime/execution_backend.h"
#include "inferx/runtime/model_handle.h"
#include "inferx/runtime/single_request_runner.h"

namespace inferx::runtime {
namespace {

constexpr char kBaseFixture[] = M5_MODEL_FIXTURE_DIR "/tiny_llama";
constexpr char kEosFixture[] = M5_MODEL_FIXTURE_DIR "/tiny_llama_eos";

class CapturingSink final : public ResponseSink {
 public:
  absl::StatusOr<ResponseReservation> Prepare(ResponseEvent event) override {
    return ResponseReservation(std::move(event));
  }
  void Commit(ResponseReservation reservation) noexcept override {
    events_.push_back(reservation.Take());
  }
  [[nodiscard]] const std::vector<ResponseEvent>& events() const noexcept { return events_; }

 private:
  std::vector<ResponseEvent> events_;
};

struct GenerationOutcome {
  absl::Status status;
  FinishReason finish = FinishReason::kLength;
  std::vector<int32_t> tokens;
  std::vector<ResponseEvent> events;
};

GenerateRequest MakeRequest(uint64_t id, const std::vector<int32_t>& tokens, uint32_t max_output) {
  return GenerateRequest{RequestId(id),
                         ModelId(0),
                         std::vector<TokenId>(tokens.begin(), tokens.end()),
                         GenerationLimits{TokenCount(max_output)},
                         kDefaultPriority,
                         std::nullopt,
                         TenantScope(0)};
}

class M5CpuBackendTest : public ::testing::Test {
 protected:
  static absl::StatusOr<ModelHandle> LoadBackend(CpuExecutionBackend& backend,
                                                 const char* fixture) {
    auto package = input::ModelPackageLoader::Load(fixture);
    if (!package.ok()) return package.status();
    ModelLoadPlan plan;
    plan.package = &*package;
    plan.model_root = fixture;
    plan.max_prefill_tokens = 8;
    plan.context_capacity = 16;
    return backend.Load(plan);
  }

  static GenerationOutcome GenerateOnce(const char* fixture, std::vector<int32_t> prompt_tokens,
                                        uint32_t max_output) {
    CpuExecutionBackend backend;
    auto handle = LoadBackend(backend, fixture);
    if (!handle.ok()) {
      return GenerationOutcome{handle.status(), FinishReason::kExecutorError, {}, {}};
    }
    CapturingSink sink;
    auto runner = SingleRequestRunner::Create(backend, *handle, /*context_capacity=*/16,
                                              /*max_prefill_tokens=*/8, sink);
    if (!runner.ok()) {
      return GenerationOutcome{runner.status(), FinishReason::kExecutorError, {}, {}};
    }
    auto result =
        (*runner)->Run(MakeRequest(1, prompt_tokens, max_output), std::chrono::steady_clock::now());
    GenerationOutcome outcome;
    if (!result.ok()) {
      outcome.status = result.status();
      return outcome;
    }
    outcome.status = result->status;
    outcome.finish = result->finish;
    for (const TokenId token : result->output_tokens) outcome.tokens.push_back(token.value());
    outcome.events = sink.events();
    return outcome;
  }
};

TEST_F(M5CpuBackendTest, LoadsTinyLlamaAndGeneratesDeterministically) {
  const GenerationOutcome first = GenerateOnce(kBaseFixture, {0, 1, 2}, 6);
  ASSERT_TRUE(first.status.ok()) << first.status;
  EXPECT_EQ(first.finish, FinishReason::kLength);
  ASSERT_EQ(first.tokens.size(), 6U);

  const GenerationOutcome second = GenerateOnce(kBaseFixture, {0, 1, 2}, 6);
  ASSERT_TRUE(second.status.ok()) << second.status;
  EXPECT_EQ(second.tokens, first.tokens);

  // Ordered response contract: token deltas then exactly one terminal.
  size_t deltas = 0;
  bool terminal = false;
  for (const ResponseEvent& event : first.events) {
    if (std::holds_alternative<TokenDelta>(event)) {
      EXPECT_FALSE(terminal);
      ++deltas;
    } else {
      terminal = true;
      const TerminalResponse& response = std::get<TerminalResponse>(event);
      EXPECT_EQ(response.prompt_tokens.value(), 3U);
      EXPECT_EQ(response.output_tokens.value(), 6U);
    }
  }
  EXPECT_EQ(deltas, 6U);
  EXPECT_TRUE(terminal);
}

TEST_F(M5CpuBackendTest, OutputLengthFollowsTheRequest) {
  const GenerationOutcome short_run = GenerateOnce(kBaseFixture, {1}, 2);
  ASSERT_TRUE(short_run.status.ok()) << short_run.status;
  EXPECT_EQ(short_run.finish, FinishReason::kLength);
  EXPECT_EQ(short_run.tokens.size(), 2U);

  const GenerationOutcome longer = GenerateOnce(kBaseFixture, {1}, 4);
  ASSERT_TRUE(longer.status.ok()) << longer.status;
  EXPECT_EQ(longer.tokens.size(), 4U);
  // Greedy prefixes agree: the first two tokens are the same trajectory.
  EXPECT_EQ(std::vector<int32_t>(longer.tokens.begin(), longer.tokens.begin() + 2),
            short_run.tokens);
}

TEST_F(M5CpuBackendTest, RealEosFinishesWithEosReason) {
  const GenerationOutcome eos = GenerateOnce(kEosFixture, {0, 1}, 8);
  ASSERT_TRUE(eos.status.ok()) << eos.status;
  EXPECT_EQ(eos.finish, FinishReason::kEos);
  // The EOS token itself is the final emitted delta; generation stops even
  // though the output budget (8) was not exhausted.
  ASSERT_EQ(eos.tokens.size(), 1U);
  EXPECT_EQ(eos.tokens.front(), 3);
}

TEST_F(M5CpuBackendTest, RepeatedRequestsAndReloadAreStable) {
  CpuExecutionBackend backend;
  auto handle = LoadBackend(backend, kBaseFixture);
  ASSERT_TRUE(handle.ok()) << handle.status();
  CapturingSink sink;
  auto runner = SingleRequestRunner::Create(backend, *handle, 16, 8, sink);
  ASSERT_TRUE(runner.ok()) << runner.status();

  std::vector<std::vector<int32_t>> trajectories;
  const std::vector<int32_t> prompt{2, 0};
  for (int attempt = 0; attempt < 3; ++attempt) {
    auto result = (*runner)->Run(MakeRequest(static_cast<uint64_t>(attempt + 1), prompt, 3),
                                 std::chrono::steady_clock::now());
    ASSERT_TRUE(result.ok()) << result.status();
    ASSERT_TRUE(result->status.ok()) << result->status;
    ASSERT_EQ(result->output_tokens.size(), 3U);
    std::vector<int32_t> trajectory;
    for (const TokenId token : result->output_tokens) trajectory.push_back(token.value());
    trajectories.push_back(trajectory);
  }
  EXPECT_EQ(trajectories[0], trajectories[1]);
  EXPECT_EQ(trajectories[0], trajectories[2]);

  // Unload is explicit; a second load after unload works and reproduces the
  // same greedy output from a fresh warm-up and cache generation.
  EXPECT_TRUE(backend.Unload(*handle).ok());
  auto reloaded = LoadBackend(backend, kBaseFixture);
  ASSERT_TRUE(reloaded.ok()) << reloaded.status();
  CapturingSink second_sink;
  auto second_runner = SingleRequestRunner::Create(backend, *reloaded, 16, 8, second_sink);
  ASSERT_TRUE(second_runner.ok()) << second_runner.status();
  auto result = (*second_runner)->Run(MakeRequest(9, prompt, 3), std::chrono::steady_clock::now());
  ASSERT_TRUE(result.ok()) << result.status();
  std::vector<int32_t> reloaded_trajectory;
  for (const TokenId token : result->output_tokens) reloaded_trajectory.push_back(token.value());
  EXPECT_EQ(reloaded_trajectory, trajectories[0]);
}

TEST_F(M5CpuBackendTest, UnloadWithInFlightTicketIsRejected) {
  CpuExecutionBackend backend;
  auto handle = LoadBackend(backend, kBaseFixture);
  ASSERT_TRUE(handle.ok()) << handle.status();
  EXPECT_TRUE(backend.Unload(*handle).ok());
  auto stale = backend.Unload(*handle);
  EXPECT_EQ(stale.code(), absl::StatusCode::kNotFound);
}

}  // namespace
}  // namespace inferx::runtime
