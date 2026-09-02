#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "inferx/artifacts/mapped_region.h"
#include "inferx/artifacts/model_locator.h"
#include "inferx/artifacts/safe_relative_path.h"
#include "src/model_adapters/artifact_tensor_adapter.h"

namespace inferx::model_adapters {
namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    std::array<char, 36> pattern{};
    constexpr std::string_view prefix = "/tmp/inferx-m4-adapter-XXXXXX";
    std::copy(prefix.begin(), prefix.end(), pattern.begin());
    const char* created = ::mkdtemp(pattern.data());
    if (created != nullptr) path_ = created;
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

TEST(M4ArtifactAdapterTest, StagesNaturalAlignmentAndMaterializesHalf) {
  TemporaryDirectory temporary;
  ASSERT_FALSE(temporary.path().empty());
  const std::array<std::byte, 3> file_bytes{std::byte{0xff}, std::byte{0x00}, std::byte{0x3c}};
  {
    std::ofstream output(temporary.path() / "weight", std::ios::binary);
    ASSERT_TRUE(output.good());
    output.write(reinterpret_cast<const char*>(file_bytes.data()),
                 static_cast<std::streamsize>(file_bytes.size()));
    ASSERT_TRUE(output.good());
  }
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok()) << session.status();
  auto relative = artifacts::SafeRelativePath::Parse("weight");
  ASSERT_TRUE(relative.ok()) << relative.status();
  auto file = session->OpenRegular(*relative);
  ASSERT_TRUE(file.ok()) << file.status();
  artifacts::ArtifactLimits limits;
  artifacts::ShardMappingPool pool(limits);
  auto lease = pool.Map(*file, artifacts::ArtifactByteRange{1, 2});
  ASSERT_TRUE(lease.ok()) << lease.status();

  model::ParameterSpec parameter{model::ParameterId(1),
                                 "weight",
                                 model::ParameterRole::kFinalNorm,
                                 {1},
                                 {artifacts::ArtifactDType::kF16},
                                 std::nullopt,
                                 std::nullopt};
  model::TensorSource source{
      *relative, file->identity(), {1, 2}, artifacts::ArtifactDType::kF16, {1}, {}};
  model::WeightPlanItem plan{model::ParameterId(1), std::move(source),
                             model::TransformSpec{model::TransformKind::kIdentity, {1}, {1}},
                             model::ShardKind::kReplicatedFullTensor};
  ArtifactTensorAdapter adapter;
  auto adapted = adapter.Adapt(plan, parameter, std::move(*lease), ByteCount(1));
  ASSERT_TRUE(adapted.ok()) << adapted.status();
  EXPECT_TRUE(adapted->staged());
  EXPECT_EQ(adapted->view().dtype(), DType::kFloat16);
  EXPECT_EQ(pool.active_regions(), 0);

  ReferenceTensorMaterializer materializer;
  auto limited = materializer.Materialize(*adapted, 0);
  EXPECT_EQ(limited.status().code(), absl::StatusCode::kResourceExhausted);
  auto materialized = materializer.Materialize(*adapted, 1);
  ASSERT_TRUE(materialized.ok()) << materialized.status();
  auto bytes = materialized->view().buffer().HostBytes();
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  ASSERT_EQ(bytes->size(), sizeof(float));
  float value = 0.0F;
  std::memcpy(&value, bytes->data(), sizeof(value));
  EXPECT_FLOAT_EQ(value, 1.0F);
  EXPECT_TRUE(materialized->Release().ok());
  EXPECT_TRUE(adapted->Release().ok());
}

}  // namespace
}  // namespace inferx::model_adapters
