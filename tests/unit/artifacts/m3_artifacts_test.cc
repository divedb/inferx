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

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/artifacts/digest.h"
#include "inferx/artifacts/mapped_region.h"
#include "inferx/artifacts/model_fingerprint.h"
#include "inferx/artifacts/model_locator.h"
#include "inferx/artifacts/safe_relative_path.h"
#include "inferx/artifacts/safetensors_reader.h"
#include "inferx/model/llama_model_factory.h"
#include "inferx/model/model_package.h"
#include "inferx/model/weight_planner.h"

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

std::vector<std::byte> SafeTensorFileWithDeclaredHeaderSize(uint64_t header_size) {
  std::vector<std::byte> file(8);
  for (size_t i = 0; i < 8; ++i) {
    file[i] = static_cast<std::byte>((header_size >> (8 * i)) & 0xFFU);
  }
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

TEST(ModelFingerprintTest, RejectsUnqualifiedTokenizerCapability) {
  artifacts::ModelFingerprintInput input;
  input.architecture = "llama";
  input.semantic_config = {std::byte{1}};
  auto fingerprint = artifacts::ModelFingerprint::Build(std::move(input));
  ASSERT_FALSE(fingerprint.ok());
  EXPECT_TRUE(absl::IsFailedPrecondition(fingerprint.status()));
}

TEST(ModelFingerprintTest, RejectsIncompleteCompatibilityPreimage) {
  constexpr std::string_view content = "artifact";
  auto digest = artifacts::HashBytes(std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(content.data()), content.size()));
  ASSERT_TRUE(digest.ok()) << digest.status();
  auto path = artifacts::SafeRelativePath::Parse("config.json");
  ASSERT_TRUE(path.ok());
  artifacts::ModelFingerprintInput input;
  input.architecture = "llama";
  input.semantic_config = {std::byte{1}};
  input.tokenizer_capability = {std::byte{2}};
  input.artifacts.push_back({*path, content.size(), *digest});
  input.rope = {std::byte{3}};
  input.source_layout.clear();
  auto fingerprint = artifacts::ModelFingerprint::Build(std::move(input));
  ASSERT_FALSE(fingerprint.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(fingerprint.status()));
}

TEST(ArtifactFileTest, IdentityChecksDoNotConsumeOpenIdsAndDetectMutation) {
  TemporaryDirectory temporary;
  WriteText(temporary.path() / "first", "first");
  WriteText(temporary.path() / "second", "second");
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok()) << session.status();
  auto first_path = artifacts::SafeRelativePath::Parse("first");
  auto second_path = artifacts::SafeRelativePath::Parse("second");
  ASSERT_TRUE(first_path.ok());
  ASSERT_TRUE(second_path.ok());
  auto first = session->OpenRegular(*first_path);
  ASSERT_TRUE(first.ok()) << first.status();
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_TRUE(first->CheckUnchanged().ok());
  }
  auto second = session->OpenRegular(*second_path);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(second->identity().internal_id, first->identity().internal_id + 1);

  WriteText(temporary.path() / "first", "first changed");
  EXPECT_TRUE(absl::IsAborted(first->CheckUnchanged()));
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

TEST(SafeTensorReaderTest, DistinguishesMalformedAndResourceLimitedHeaderLengths) {
  TemporaryDirectory temporary;
  WriteBytes(temporary.path() / "empty-header.safetensors",
             SafeTensorFileWithDeclaredHeaderSize(0));
  WriteBytes(temporary.path() / "large-header.safetensors",
             SafeTensorFileWithDeclaredHeaderSize(1024));
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok()) << session.status();
  artifacts::SafeTensorReader reader;

  auto empty_path = artifacts::SafeRelativePath::Parse("empty-header.safetensors");
  ASSERT_TRUE(empty_path.ok());
  auto empty = session->OpenRegular(*empty_path);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(absl::IsDataLoss(reader.ReadHeader(*empty).status()));

  auto large_path = artifacts::SafeRelativePath::Parse("large-header.safetensors");
  ASSERT_TRUE(large_path.ok());
  auto large = session->OpenRegular(*large_path);
  ASSERT_TRUE(large.ok()) << large.status();
  artifacts::ArtifactLimits limits;
  limits.max_safetensors_header_bytes = 128;
  EXPECT_TRUE(absl::IsResourceExhausted(reader.ReadHeader(*large, limits).status()));
}

TEST(SafeTensorReaderTest, AcceptsScalarSubByteAndZeroByteBoundaryLayouts) {
  TemporaryDirectory temporary;
  const auto bytes = SafeTensorFile(
      R"({"empty":{"dtype":"F16","shape":[0,7],"data_offsets":[0,0]},"packed":{"dtype":"F4","shape":[3],"data_offsets":[0,2]},"scalar":{"dtype":"F32","shape":[],"data_offsets":[2,6]},"tail":{"dtype":"U8","shape":[0],"data_offsets":[6,6]}})",
      6);
  WriteBytes(temporary.path() / "edge.safetensors", bytes);
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok()) << session.status();
  auto path = artifacts::SafeRelativePath::Parse("edge.safetensors");
  ASSERT_TRUE(path.ok());
  auto file = session->OpenRegular(*path);
  ASSERT_TRUE(file.ok()) << file.status();
  artifacts::SafeTensorReader reader;
  auto metadata = reader.ReadHeader(*file);
  ASSERT_TRUE(metadata.ok()) << metadata.status();
  ASSERT_EQ(metadata->tensors.size(), 4);
  ASSERT_NE(metadata->Find("empty"), nullptr);
  ASSERT_NE(metadata->Find("packed"), nullptr);
  ASSERT_NE(metadata->Find("scalar"), nullptr);
  ASSERT_NE(metadata->Find("tail"), nullptr);
  EXPECT_EQ(metadata->Find("empty")->packed_size_bytes, 0);
  EXPECT_EQ(metadata->Find("packed")->packed_size_bytes, 2);
  EXPECT_EQ(metadata->Find("scalar")->packed_size_bytes, 4);
  EXPECT_EQ(metadata->Find("tail")->data.offset, 6);
}

TEST(SafeTensorReaderTest, EnforcesTensorCountWhenMetadataIsPresent) {
  TemporaryDirectory temporary;
  const auto bytes = SafeTensorFile(
      R"({"__metadata__":{"format":"pt"},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})",
      2);
  WriteBytes(temporary.path() / "many.safetensors", bytes);
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok()) << session.status();
  auto path = artifacts::SafeRelativePath::Parse("many.safetensors");
  ASSERT_TRUE(path.ok());
  auto file = session->OpenRegular(*path);
  ASSERT_TRUE(file.ok()) << file.status();
  artifacts::ArtifactLimits limits;
  limits.max_tensors = 1;
  artifacts::SafeTensorReader reader;
  auto metadata = reader.ReadHeader(*file, limits);
  ASSERT_FALSE(metadata.ok());
  EXPECT_TRUE(absl::IsResourceExhausted(metadata.status()));
}

TEST(MappedTensorReaderTest, StreamsExactWindowsFromOwnedFileDescriptor) {
  TemporaryDirectory temporary;
  WriteText(temporary.path() / "weights", "0123456789");
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok()) << session.status();
  auto path = artifacts::SafeRelativePath::Parse("weights");
  ASSERT_TRUE(path.ok());
  auto file = session->OpenRegular(*path);
  ASSERT_TRUE(file.ok()) << file.status();

  artifacts::ArtifactLimits limits;
  limits.map_window_bytes = 4;
  artifacts::ShardMappingPool pool(limits);
  auto reader = pool.OpenReader(*file, {1, 9});
  ASSERT_TRUE(reader.ok()) << reader.status();
  file = artifacts::ArtifactFile();

  std::string observed;
  std::vector<size_t> window_sizes;
  while (true) {
    auto window = reader->Next();
    ASSERT_TRUE(window.ok()) << window.status();
    if (!window->has_value()) break;
    auto lease = std::move(*window).value_or(artifacts::MappedTensorLease{});
    const auto bytes = lease.bytes();
    window_sizes.push_back(bytes.size());
    observed.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    EXPECT_EQ(pool.active_regions(), 1);
  }
  EXPECT_EQ(observed, "123456789");
  EXPECT_EQ(window_sizes, (std::vector<size_t>{4, 4, 1}));
  EXPECT_EQ(reader->consumed_bytes(), 9);
  EXPECT_EQ(reader->remaining_bytes(), 0);
  EXPECT_EQ(pool.active_regions(), 0);
  EXPECT_EQ(pool.active_mapped_bytes(), 0);
}

TEST(MappedTensorReaderTest, DoesNotAdvanceWhenPoolBudgetIsExhausted) {
  TemporaryDirectory temporary;
  WriteText(temporary.path() / "weights", "abcdefgh");
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok()) << session.status();
  auto path = artifacts::SafeRelativePath::Parse("weights");
  ASSERT_TRUE(path.ok());
  auto file = session->OpenRegular(*path);
  ASSERT_TRUE(file.ok()) << file.status();

  artifacts::ArtifactLimits limits;
  limits.map_window_bytes = 4;
  limits.max_active_mappings = 1;
  artifacts::ShardMappingPool pool(limits);
  auto reader = pool.OpenReader(*file, {0, 8});
  ASSERT_TRUE(reader.ok()) << reader.status();

  auto first = reader->Next();
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(first->has_value());
  EXPECT_EQ(reader->consumed_bytes(), 4);
  auto exhausted = reader->Next();
  ASSERT_FALSE(exhausted.ok());
  EXPECT_TRUE(absl::IsResourceExhausted(exhausted.status()));
  EXPECT_EQ(reader->consumed_bytes(), 4);

  first->reset();
  auto second = reader->Next();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(second->has_value());
  EXPECT_EQ(reader->consumed_bytes(), 8);
  auto second_lease = std::move(*second).value_or(artifacts::MappedTensorLease{});
  EXPECT_EQ(second_lease.requested_range(), (artifacts::ArtifactByteRange{4, 4}));
}

TEST(MappedTensorReaderTest, ShrinksUnalignedWindowToFitRoundedByteBudget) {
  const long page_result = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_result, 0);
  const uint64_t page = static_cast<uint64_t>(page_result);
  TemporaryDirectory temporary;
  WriteText(temporary.path() / "weights", std::string(static_cast<size_t>(page + 2), 'x'));
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok()) << session.status();
  auto path = artifacts::SafeRelativePath::Parse("weights");
  ASSERT_TRUE(path.ok());
  auto file = session->OpenRegular(*path);
  ASSERT_TRUE(file.ok()) << file.status();

  artifacts::ArtifactLimits limits;
  limits.map_window_bytes = page;
  limits.max_mapped_bytes = page;
  artifacts::ShardMappingPool pool(limits);
  auto reader = pool.OpenReader(*file, {1, page + 1});
  ASSERT_TRUE(reader.ok()) << reader.status();
  auto first = reader->Next();
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(first->has_value());
  auto first_lease = std::move(*first).value_or(artifacts::MappedTensorLease{});
  EXPECT_EQ(first_lease.requested_range(), (artifacts::ArtifactByteRange{1, page - 1}));
  first_lease = artifacts::MappedTensorLease{};
  auto second = reader->Next();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(second->has_value());
  auto second_lease = std::move(*second).value_or(artifacts::MappedTensorLease{});
  EXPECT_EQ(second_lease.requested_range(), (artifacts::ArtifactByteRange{page, 2}));
  EXPECT_EQ(reader->remaining_bytes(), 0);
}

TEST(ExternalTensorCatalogTest, RejectsRangeOutsideOpenedShard) {
  auto shard = artifacts::SafeRelativePath::Parse("model.safetensors");
  ASSERT_TRUE(shard.ok());
  artifacts::FileIdentity identity;
  identity.size = 4;
  artifacts::ArtifactTensor tensor{"weight", artifacts::ArtifactDType::kF16, {1}, {0, 2}, 3, 2};
  std::vector<model::ExternalTensor> tensors;
  tensors.push_back({"weight", *shard, identity, std::move(tensor), {}, std::nullopt});
  auto catalog = model::ExternalTensorCatalog::Build(std::move(tensors));
  ASSERT_FALSE(catalog.ok());
  EXPECT_TRUE(absl::IsDataLoss(catalog.status()));
}

TEST(WeightPlannerTest, RejectsDuplicateParameterIdsBeforeCoverage) {
  model::LlamaSpec llama;
  model::ModelSpec spec(std::move(llama));
  const std::vector<artifacts::ArtifactDType> dtypes = {artifacts::ArtifactDType::kF16};
  model::ParameterCatalog parameters;
  parameters.push_back({model::ParameterId(7),
                        "first",
                        model::ParameterRole::kFinalNorm,
                        {1},
                        dtypes,
                        std::nullopt,
                        std::nullopt});
  parameters.push_back({model::ParameterId(7),
                        "second",
                        model::ParameterRole::kFinalNorm,
                        {1},
                        dtypes,
                        std::nullopt,
                        std::nullopt});
  auto external = model::ExternalTensorCatalog::Build({});
  ASSERT_TRUE(external.ok()) << external.status();
  model::WeightPlanner planner;
  auto plan = planner.Build(spec, parameters, *external);
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(absl::IsInternal(plan.status()));
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

  std::ostringstream manifest;
  manifest
      << R"({"schema_version":1,"model_revision":"tiny-v1","weights_entry":"model.safetensors","files":[)";
  size_t manifest_entry = 0;
  for (const std::string_view name : {"config.json", "model.safetensors", "tokenizer.json"}) {
    auto path = artifacts::SafeRelativePath::Parse(name);
    ASSERT_TRUE(path.ok());
    auto file = session->OpenRegular(*path);
    ASSERT_TRUE(file.ok()) << file.status();
    auto digest = artifacts::HashFile(*file);
    ASSERT_TRUE(digest.ok()) << digest.status();
    if (manifest_entry++ != 0) manifest << ',';
    manifest << R"({"path":")" << name << R"(","size":)" << file->identity().size
             << R"(,"blake3":")" << digest->Hex() << R"("})";
  }
  manifest << "]}";
  WriteText(temporary.path() / "inferx.manifest.json", manifest.str());
  auto verified = loader.Inspect(temporary.path());
  ASSERT_TRUE(verified.ok()) << verified.status();
  EXPECT_TRUE(verified->integrity_manifest);
  EXPECT_EQ(verified->model_revision, "tiny-v1");

  WriteText(temporary.path() / "tokenizer.json", "[]");
  auto mismatch = loader.Inspect(temporary.path());
  ASSERT_FALSE(mismatch.ok());
  EXPECT_TRUE(absl::IsDataLoss(mismatch.status()));
  WriteText(temporary.path() / "tokenizer.json", "{}");

  WriteText(temporary.path() / "tokenizer_config.json", std::string(2048, 'x'));
  artifacts::ArtifactLimits limits;
  limits.max_json_bytes = 1024;
  auto oversized_optional = loader.Inspect(temporary.path(), limits);
  ASSERT_FALSE(oversized_optional.ok());
  EXPECT_TRUE(absl::IsResourceExhausted(oversized_optional.status()));
  EXPECT_NE(oversized_optional.status().message().find("tokenizer_config.json"),
            std::string_view::npos);
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

TEST(ModelArtifactLoaderTest, BuildsCompleteShardedWeightPlan) {
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
  ASSERT_TRUE(session.ok()) << session.status();
  auto config_path = artifacts::SafeRelativePath::Parse("config.json");
  ASSERT_TRUE(config_path.ok());
  auto config_file = session->OpenRegular(*config_path);
  ASSERT_TRUE(config_file.ok()) << config_file.status();
  model::LlamaModelFactory factory;
  auto spec = factory.ParseConfig(*config_file);
  ASSERT_TRUE(spec.ok()) << spec.status();
  auto parameters = factory.BuildParameters(*spec);
  ASSERT_TRUE(parameters.ok()) << parameters.status();

  std::array<std::ostringstream, 2> headers;
  headers[0] << '{';
  headers[1] << '{';
  std::array<uint64_t, 2> offsets{};
  std::array<size_t, 2> emitted{};
  std::ostringstream weight_map;
  weight_map << '{';
  uint64_t total_payload = 0;
  for (size_t i = 0; i < parameters->size(); ++i) {
    const auto& parameter = (*parameters)[i];
    const size_t shard = i % 2;
    auto packed = artifacts::PackedTensorBytes(artifacts::ArtifactDType::kF16, parameter.shape);
    ASSERT_TRUE(packed.ok()) << packed.status();
    if (emitted[shard]++ != 0) headers[shard] << ',';
    headers[shard] << '"' << parameter.canonical_name
                   << "\":{\"dtype\":\"F16\",\"shape\":" << ShapeJson(parameter.shape)
                   << ",\"data_offsets\":[" << offsets[shard] << ',' << offsets[shard] + *packed
                   << "]}";
    offsets[shard] += *packed;
    total_payload += *packed;
    if (i != 0) weight_map << ',';
    weight_map << '"' << parameter.canonical_name << "\":\"model-0000" << shard + 1
               << "-of-00002.safetensors\"";
  }
  headers[0] << '}';
  headers[1] << '}';
  weight_map << '}';
  WriteBytes(temporary.path() / "model-00001-of-00002.safetensors",
             SafeTensorFile(headers[0].str(), static_cast<size_t>(offsets[0])));
  WriteBytes(temporary.path() / "model-00002-of-00002.safetensors",
             SafeTensorFile(headers[1].str(), static_cast<size_t>(offsets[1])));
  std::ostringstream index;
  index << R"({"metadata":{"total_size":)" << total_payload << R"(},"weight_map":)"
        << weight_map.str() << '}';
  WriteText(temporary.path() / "model.safetensors.index.json", index.str());

  model::ModelArtifactLoader loader;
  auto package = loader.Inspect(temporary.path());
  ASSERT_TRUE(package.ok()) << package.status();
  EXPECT_EQ(package->weights_entry.string(), "model.safetensors.index.json");
  EXPECT_EQ(package->weight_plan.coverage.expected, 12);
  EXPECT_EQ(package->weight_plan.coverage.assigned, 12);
  EXPECT_EQ(package->weight_plan.coverage.external, 12);
  EXPECT_EQ(package->artifacts.size(), 5);
  std::set<std::string> planned_shards;
  for (const auto& item : package->weight_plan.items) {
    planned_shards.insert(item.source.shard.string());
  }
  EXPECT_EQ(planned_shards, (std::set<std::string>{"model-00001-of-00002.safetensors",
                                                   "model-00002-of-00002.safetensors"}));
}

}  // namespace
}  // namespace inferx
