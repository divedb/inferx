// The M3 tokenizer differential gate (m3.md section 18; ADR 0024): the
// committed 10,000-case corpus produced by the pinned Hugging Face Python
// oracle must match exactly -- encode with and without special tokens,
// one-shot decode, and streaming decode equality.
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/base/id.h"
#include "inferx/tokenization/tokenizer.h"
#include "simdjson.h"

namespace inferx::tokenization {
namespace {

struct Case {
  std::string text;
  std::vector<TokenId> ids_special;
  std::vector<TokenId> ids_plain;
  std::string decoded_plain;
};

std::vector<Case> LoadCases(const char* path) {
  std::vector<Case> cases;
  std::ifstream stream(path);
  if (!stream.good()) {
    ADD_FAILURE() << "cannot open " << path;
    return cases;
  }
  simdjson::ondemand::parser parser;
  std::string line;
  size_t lineno = 0;
  while (std::getline(stream, line)) {
    ++lineno;
    simdjson::ondemand::document doc = parser.iterate(line);
    simdjson::ondemand::object obj = doc.get_object();
    Case record;
    simdjson::ondemand::value text = obj.find_field("text");
    record.text = std::string(std::string_view(text));
    simdjson::ondemand::array special = obj.find_field("ids_special");
    for (int64_t id : special) {
      record.ids_special.push_back(TokenId(static_cast<int32_t>(id)));
    }
    simdjson::ondemand::array plain = obj.find_field("ids_plain");
    for (int64_t id : plain) {
      record.ids_plain.push_back(TokenId(static_cast<int32_t>(id)));
    }
    simdjson::ondemand::value decoded = obj.find_field("decoded_plain");
    record.decoded_plain = std::string(std::string_view(decoded));
    cases.push_back(std::move(record));
  }
  return cases;
}

bool ValidUtf8(const std::string& text) {
  // Decode-and-reencode through the engine is not available here; a plain
  // UTF-8 validation of the chunk.
  const auto* p = reinterpret_cast<const unsigned char*>(text.data());
  const auto* end = p + text.size();
  while (p < end) {
    const unsigned char c = *p;
    size_t extra;
    uint32_t code;
    if (c < 0x80) {
      ++p;
      continue;
    }
    if ((c & 0xE0) == 0xC0) {
      extra = 1;
      code = c & 0x1F;
    } else if ((c & 0xF0) == 0xE0) {
      extra = 2;
      code = c & 0x0F;
    } else if ((c & 0xF8) == 0xF0) {
      extra = 3;
      code = c & 0x07;
    } else {
      return false;
    }
    if (p + extra >= end) return false;
    for (size_t i = 1; i <= extra; ++i) {
      const unsigned char cc = p[i];
      if ((cc & 0xC0) != 0x80) return false;
      code = (code << 6) | (cc & 0x3F);
    }
    if (extra == 1 && code < 0x80) return false;
    if (extra == 2 && code < 0x800) return false;
    if (extra == 3 && code < 0x10000) return false;
    if (code > 0x10FFFF) return false;
    if (code >= 0xD800 && code <= 0xDFFF) return false;
    p += extra + 1;
  }
  return true;
}

class TokenizerDifferentialTest : public ::testing::TestWithParam<const char*> {};

TEST_P(TokenizerDifferentialTest, MatchesOracleExactly) {
  const std::string directory = GetParam();
  std::ifstream json((directory + "/tokenizer.json").c_str(), std::ios::binary);
  std::ostringstream buffer;
  buffer << json.rdbuf();
  TokenizerArtifacts artifacts;
  artifacts.tokenizer_json = buffer.str();
  std::ifstream config((directory + "/tokenizer_config.json").c_str(), std::ios::binary);
  if (config.good()) {
    std::ostringstream cfg;
    cfg << config.rdbuf();
    artifacts.tokenizer_config = cfg.str();
  }

  auto loaded = Tokenizer::Load(std::move(artifacts));
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  const Tokenizer& tokenizer = **loaded;

  auto cases_result = LoadCases((directory + "/cases.jsonl").c_str());
  ASSERT_FALSE(cases_result.empty());

  EncodeOptions with_special;
  EncodeOptions without_special;
  without_special.add_special_tokens = false;
  DecodeOptions decode;  // skip_special_tokens=false, cleanup=false

  size_t streaming_checked = 0;
  for (size_t i = 0; i < cases_result.size(); ++i) {
    const Case& record = cases_result[i];
    ASSERT_NO_FATAL_FAILURE({
      auto encoded_special = tokenizer.Encode(record.text, with_special);
      ASSERT_TRUE(encoded_special.ok()) << i << ": " << encoded_special.status();
      ASSERT_EQ(*encoded_special, record.ids_special)
          << i << ": add_special_tokens=true diverged for " << record.text;

      auto encoded_plain = tokenizer.Encode(record.text, without_special);
      ASSERT_TRUE(encoded_plain.ok()) << i << ": " << encoded_plain.status();
      ASSERT_EQ(*encoded_plain, record.ids_plain)
          << i << ": add_special_tokens=false diverged for " << record.text;

      auto decoded = tokenizer.Decode(record.ids_plain, decode);
      ASSERT_TRUE(decoded.ok()) << i << ": " << decoded.status();
      ASSERT_EQ(*decoded, record.decoded_plain) << i << ": decode diverged for " << record.text;
    });

    // Streaming equality on every 5th case; per-chunk UTF-8 validity on
    // every 25th (m3.md 13.4: every nonempty chunk is valid UTF-8 and prior
    // bytes never change).
    if (i % 5 == 0) {
      auto decoder = tokenizer.NewIncrementalDecoder(decode);
      ASSERT_TRUE(decoder.ok()) << decoder.status();
      std::string streamed;
      for (TokenId id : record.ids_plain) {
        auto chunk = (*decoder)->Push(id);
        ASSERT_TRUE(chunk.ok()) << i << ": " << chunk.status();
        if (chunk->has_value()) {
          if (i % 25 == 0) {
            ASSERT_TRUE(ValidUtf8(**chunk)) << i << ": invalid UTF-8 chunk in " << record.text;
          }
          streamed.append(**chunk);
        }
      }
      auto rest = (*decoder)->Finish();
      ASSERT_TRUE(rest.ok()) << i << ": " << rest.status();
      streamed.append(*rest);
      ASSERT_EQ(streamed, record.decoded_plain)
          << i << ": streaming decode diverged for " << record.text;
      ++streaming_checked;
    }
  }
  ASSERT_EQ(cases_result.size(), 2500u);
  EXPECT_GE(streaming_checked, 500u);
}

INSTANTIATE_TEST_SUITE_P(Checkpoints, TokenizerDifferentialTest,
                         ::testing::Values(TOKENIZER_REFERENCE_DIR "/qwen2.5",
                                           TOKENIZER_REFERENCE_DIR "/tinyllama",
                                           TOKENIZER_REFERENCE_DIR "/bert-base",
                                           TOKENIZER_REFERENCE_DIR "/t5"),
                         [](const ::testing::TestParamInfo<const char*>& param) {
                           std::string name = param.param;
                           const size_t slash = name.find_last_of('/');
                           if (slash != std::string::npos) name = name.substr(slash + 1);
                           for (char& c : name) {
                             if (c == '.' || c == '-') c = '_';
                           }
                           return name;
                         });

}  // namespace
}  // namespace inferx::tokenization
