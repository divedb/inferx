// Smoke: load a real tokenizer.json through the qualified facade, encode,
// decode, stream-decode, and dump metadata. Fixture paths are passed by the
// CMake test definition; missing fixtures skip (they are downloaded
// fixtures, not committed ones).
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/tokenization/tokenizer.h"

namespace inferx::tokenization {
namespace {

std::string ReadFile(const char* path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

TEST(QualifiedTokenizerSmoke, EncodeDecodeStream) {
  const std::string tokenizer_json = ReadFile(TOKENIZER_JSON_PATH);
  ASSERT_FALSE(tokenizer_json.empty()) << "fixture missing";
  const std::string config_json = ReadFile(TOKENIZER_CONFIG_PATH);

  TokenizerArtifacts artifacts;
  artifacts.tokenizer_json = tokenizer_json;
  if (!config_json.empty()) artifacts.tokenizer_config = config_json;

  auto loaded = Tokenizer::Load(std::move(artifacts));
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  const Tokenizer& tokenizer = **loaded;

  const TokenizerMetadata& metadata = tokenizer.metadata();
  EXPECT_GT(metadata.total_vocab_size, 100000u);
  EXPECT_FALSE(metadata.model_type.empty());
  EXPECT_TRUE(metadata.has_special_tokens);

  // Round trip: encode then decode equals the text back (Qwen byte-level
  // BPE with add_special_tokens default; plain ASCII round trips exactly).
  const std::string text = "Hello, InferX tokenizer!";
  auto ids = tokenizer.Encode(text, EncodeOptions{});
  ASSERT_TRUE(ids.ok()) << ids.status();
  ASSERT_FALSE(ids->empty());

  DecodeOptions decode;
  decode.skip_special_tokens = true;
  auto decoded = tokenizer.Decode(*ids, decode);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, text);

  // Streaming: chunks concatenated + Finish == one-shot decode.
  auto decoder = tokenizer.NewIncrementalDecoder(decode);
  ASSERT_TRUE(decoder.ok()) << decoder.status();
  std::string streamed;
  for (TokenId id : *ids) {
    auto chunk = (*decoder)->Push(id);
    ASSERT_TRUE(chunk.ok()) << chunk.status();
    if (chunk->has_value()) streamed.append(**chunk);
  }
  auto finish = (*decoder)->Finish();
  ASSERT_TRUE(finish.ok()) << finish.status();
  streamed.append(*finish);
  EXPECT_EQ(streamed, *decoded);

  // Metadata canonical record is stable across calls.
  EXPECT_EQ(CanonicalMetadataRecord(metadata), CanonicalMetadataRecord(metadata));
  EXPECT_FALSE(CanonicalMetadataRecord(metadata).empty());
}

TEST(QualifiedTokenizerSmoke, MalformedJsonIsAnErrorNotAnAbort) {
  TokenizerArtifacts artifacts;
  artifacts.tokenizer_json = "{\"truncated\": ";
  auto loaded = Tokenizer::Load(std::move(artifacts));
  EXPECT_FALSE(loaded.ok());

  TokenizerArtifacts not_a_tokenizer;
  not_a_tokenizer.tokenizer_json = "{\"model\": null}";
  auto rejected = Tokenizer::Load(std::move(not_a_tokenizer));
  EXPECT_FALSE(rejected.ok());
}

}  // namespace
}  // namespace inferx::tokenization
