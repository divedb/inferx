#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/artifacts/digest.h"
#include "inferx/artifacts/mapped_region.h"
#include "inferx/artifacts/model_fingerprint.h"
#include "inferx/artifacts/model_locator.h"
#include "inferx/artifacts/safe_relative_path.h"
#include "inferx/artifacts/safetensors_reader.h"
#include "inferx/model/llama_model_factory.h"
#include "inferx/model/model_package.h"

namespace inferx {
namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    std::array<char, 32> pattern{};
    constexpr std::string_view prefix = "/tmp/inferx-m3-test-XXXXXX";
    std::copy(prefix.begin(), prefix.end(), pattern.begin());
    path_ = ::mkdtemp(pattern.data());
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void WriteBytes(const std::filesystem::path& path, std::span<const std::byte> bytes) {
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output.good());
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(output.good());
}

void WriteText(const std::filesystem::path& path, std::string_view text) {
  WriteBytes(path, std::span<const std::byte>(reinterpret_cast<const std::byte*>(text.data()),
                                              text.size()));
}

std::vector<std::byte> SafeTensorFile(std::string header, size_t payload_size) {
  while (header.size() % 8 != 0) header.push_back(' ');
  std::vector<std::byte> file(8 + header.size() + payload_size);
  const uint64_t size = header.size();
  for (size_t i = 0; i < 8; ++i) {
    file[i] = static_cast<std::byte>((size >> (8 * i)) & 0xFFU);
  }
  std::copy(reinterpret_cast<const std::byte*>(header.data()),
            reinterpret_cast<const std::byte*>(header.data() + header.size()), file.begin() + 8);
  return file;
}

std::string ShapeJson(const artifacts::ArtifactShape& shape) {
  std::ostringstream output;
  output << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) output << ',';
    output << shape[i];
  }
  output << ']';
  return output.str();
}

TEST(SafeRelativePathTest, RejectsTraversalAndAliases) {
  EXPECT_FALSE(artifacts::SafeRelativePath::Parse("").ok());
  EXPECT_FALSE(artifacts::SafeRelativePath::Parse("/absolute").ok());
  EXPECT_FALSE(artifacts::SafeRelativePath::Parse("../escape").ok());
  EXPECT_FALSE(artifacts::SafeRelativePath::Parse("a/./b").ok());
  EXPECT_FALSE(artifacts::SafeRelativePath::Parse("a//b").ok());
  EXPECT_FALSE(artifacts::SafeRelativePath::Parse("a\\b").ok());
  EXPECT_TRUE(artifacts::SafeRelativePath::Parse("shards/model-1.safetensors").ok());
}

TEST(DigestTest, MatchesOfficialBlake3Vectors) {
  auto empty = artifacts::HashBytes({});
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(empty->Hex(), "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262");
  constexpr std::string_view abc = "abc";
  auto digest = artifacts::HashBytes(
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(abc.data()), abc.size()));
  ASSERT_TRUE(digest.ok()) << digest.status();
  EXPECT_EQ(digest->Hex(), "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85");
}

TEST(ModelFingerprintTest, IsDeterministicAndDomainSeparated) {
  constexpr std::string_view content = "artifact";
  auto digest = artifacts::HashBytes(std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(content.data()), content.size()));
  ASSERT_TRUE(digest.ok());
  auto path = artifacts::SafeRelativePath::Parse("config.json");
  ASSERT_TRUE(path.ok());
  artifacts::ModelFingerprintInput input;
  input.architecture = "llama";
  input.semantic_config = {std::byte{1}, std::byte{2}};
  input.tokenizer_capability = {std::byte{3}};
  input.artifacts.push_back({*path, content.size(), *digest});
  input.rope = {std::byte{4}};
  auto first = artifacts::ModelFingerprint::Build(input);
  auto second = artifacts::ModelFingerprint::Build(input);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(first->digest(), second->digest());

  input.model_revision = "different";
  auto changed = artifacts::ModelFingerprint::Build(std::move(input));
  ASSERT_TRUE(changed.ok());
  EXPECT_NE(first->digest(), changed->digest());
}

TEST(ModelLocatorTest, RejectsDescendantSymlink) {
  TemporaryDirectory temporary;
  WriteText(temporary.path() / "outside", "secret");
  std::filesystem::create_symlink(temporary.path() / "outside", temporary.path() / "linked");
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok()) << session.status();
  auto linked = artifacts::SafeRelativePath::Parse("linked");
  ASSERT_TRUE(linked.ok());
  EXPECT_FALSE(session->OpenRegular(*linked).ok());
}

TEST(SafeTensorReaderTest, ValidatesShapeOffsetsAndMappingLifetime) {
  TemporaryDirectory temporary;
  const auto bytes =
      SafeTensorFile(R"({"weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}})", 8);
  WriteBytes(temporary.path() / "model.safetensors", bytes);
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok()) << session.status();
  auto path = artifacts::SafeRelativePath::Parse("model.safetensors");
  ASSERT_TRUE(path.ok());
  auto file = session->OpenRegular(*path);
  ASSERT_TRUE(file.ok()) << file.status();
  artifacts::SafeTensorReader reader;
  auto metadata = reader.ReadHeader(*file);
  ASSERT_TRUE(metadata.ok()) << metadata.status();
  ASSERT_EQ(metadata->tensors.size(), 1);
  EXPECT_EQ(metadata->tensors[0].packed_size_bytes, 8);
  EXPECT_EQ(metadata->tensors[0].shape, (artifacts::ArtifactShape{2, 2}));

  artifacts::ShardMappingPool pool({});
  {
    auto lease = pool.Map(*file, {metadata->tensors[0].absolute_file_offset, 8});
    ASSERT_TRUE(lease.ok()) << lease.status();
    EXPECT_EQ(lease->bytes().size(), 8);
    EXPECT_EQ(pool.active_regions(), 1);
  }
  EXPECT_EQ(pool.active_regions(), 0);
  EXPECT_EQ(pool.active_mapped_bytes(), 0);
}

TEST(SafeTensorReaderTest, RejectsPayloadHoles) {
  TemporaryDirectory temporary;
  const auto bytes = SafeTensorFile(R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})", 2);
  WriteBytes(temporary.path() / "bad.safetensors", bytes);
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok());
  auto path = artifacts::SafeRelativePath::Parse("bad.safetensors");
  ASSERT_TRUE(path.ok());
  auto file = session->OpenRegular(*path);
  ASSERT_TRUE(file.ok());
  artifacts::SafeTensorReader reader;
  EXPECT_FALSE(reader.ReadHeader(*file).ok());
}

TEST(ModelArtifactLoaderTest, BuildsCompleteTinyLlamaWeightPlan) {
  TemporaryDirectory temporary;
  constexpr std::string_view config = R"({
    "architectures":["LlamaForCausalLM"],
    "model_type":"llama",
    "vocab_size":4,
    "hidden_size":2,
    "intermediate_size":4,
    "num_hidden_layers":1,
    "num_attention_heads":1,
    "num_key_value_heads":1,
    "max_position_embeddings":16,
    "rms_norm_eps":0.00001,
    "tie_word_embeddings":false
  })";
  WriteText(temporary.path() / "config.json", config);
  WriteText(temporary.path() / "tokenizer.json", "{}");

  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok());
  auto config_path = artifacts::SafeRelativePath::Parse("config.json");
  ASSERT_TRUE(config_path.ok());
  auto config_file = session->OpenRegular(*config_path);
  ASSERT_TRUE(config_file.ok());
  model::LlamaModelFactory factory;
  auto spec = factory.ParseConfig(*config_file);
  ASSERT_TRUE(spec.ok()) << spec.status();
  auto parameters = factory.BuildParameters(*spec);
  ASSERT_TRUE(parameters.ok()) << parameters.status();

  uint64_t offset = 0;
  std::ostringstream header;
  header << '{';
  for (size_t i = 0; i < parameters->size(); ++i) {
    const auto& parameter = (*parameters)[i];
    auto packed = artifacts::PackedTensorBytes(artifacts::ArtifactDType::kF16, parameter.shape);
    ASSERT_TRUE(packed.ok());
    if (i != 0) header << ',';
    header << '"' << parameter.canonical_name
           << "\":{\"dtype\":\"F16\",\"shape\":" << ShapeJson(parameter.shape)
           << ",\"data_offsets\":[" << offset << ',' << offset + *packed << "]}";
    offset += *packed;
  }
  header << '}';
  WriteBytes(temporary.path() / "model.safetensors",
             SafeTensorFile(header.str(), static_cast<size_t>(offset)));

  model::ModelArtifactLoader loader;
  auto package = loader.Inspect(temporary.path());
  ASSERT_TRUE(package.ok()) << package.status();
  EXPECT_EQ(package->weight_plan.coverage.expected, 12);
  EXPECT_EQ(package->weight_plan.coverage.assigned, 12);
  EXPECT_EQ(package->weight_plan.coverage.aliased, 0);
  EXPECT_FALSE(package->integrity_manifest);
}

TEST(ModelArtifactLoaderTest, NormalizesOmittedTiedLmHeadToAlias) {
  TemporaryDirectory temporary;
  constexpr std::string_view config = R"({
    "architectures":["LlamaForCausalLM"],
    "model_type":"llama",
    "vocab_size":4,
    "hidden_size":2,
    "intermediate_size":4,
    "num_hidden_layers":1,
    "num_attention_heads":1,
    "max_position_embeddings":16,
    "rms_norm_eps":0.00001,
    "tie_word_embeddings":true
  })";
  WriteText(temporary.path() / "config.json", config);
  WriteText(temporary.path() / "tokenizer.json", "{}");

  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok());
  auto config_path = artifacts::SafeRelativePath::Parse("config.json");
  ASSERT_TRUE(config_path.ok());
  auto config_file = session->OpenRegular(*config_path);
  ASSERT_TRUE(config_file.ok());
  model::LlamaModelFactory factory;
  auto spec = factory.ParseConfig(*config_file);
  ASSERT_TRUE(spec.ok()) << spec.status();
  auto parameters = factory.BuildParameters(*spec);
  ASSERT_TRUE(parameters.ok()) << parameters.status();

  uint64_t offset = 0;
  size_t emitted = 0;
  std::ostringstream header;
  header << '{';
  for (const auto& parameter : *parameters) {
    if (parameter.alias_target.has_value()) continue;
    auto packed = artifacts::PackedTensorBytes(artifacts::ArtifactDType::kF16, parameter.shape);
    ASSERT_TRUE(packed.ok());
    if (emitted++ != 0) header << ',';
    header << '"' << parameter.canonical_name
           << "\":{\"dtype\":\"F16\",\"shape\":" << ShapeJson(parameter.shape)
           << ",\"data_offsets\":[" << offset << ',' << offset + *packed << "]}";
    offset += *packed;
  }
  header << '}';
  WriteBytes(temporary.path() / "model.safetensors",
             SafeTensorFile(header.str(), static_cast<size_t>(offset)));

  model::ModelArtifactLoader loader;
  auto package = loader.Inspect(temporary.path());
  ASSERT_TRUE(package.ok()) << package.status();
  EXPECT_EQ(package->weight_plan.coverage.expected, 12);
  EXPECT_EQ(package->weight_plan.coverage.assigned, 11);
  EXPECT_EQ(package->weight_plan.coverage.aliased, 1);
}

}  // namespace
}  // namespace inferx
