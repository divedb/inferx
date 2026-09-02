#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <set>

#include "gtest/gtest.h"
#include "inferx/artifacts/mapped_region.h"
#include "inferx/artifacts/model_locator.h"
#include "inferx/model/model_package.h"

namespace inferx {
namespace {

TEST(ModelPackageIntegrationTest, MapsEveryParameterFromCommittedShardedFixtureExactlyOnce) {
  constexpr uint64_t kMappingLimit = 64ULL * 1024;
  const std::filesystem::path fixture =
      std::filesystem::path(INFERX_ARTIFACT_FIXTURE_DIR) / "tiny_llama";
  model::ModelArtifactLoader loader;
  auto package = loader.Inspect(fixture);
  ASSERT_TRUE(package.ok()) << package.status();
  EXPECT_EQ(package->weights_entry.string(), "model.safetensors.index.json");
  EXPECT_EQ(package->weight_plan.coverage.expected, 12);
  EXPECT_EQ(package->weight_plan.coverage.assigned, 12);
  EXPECT_EQ(package->weight_plan.coverage.aliased, 0);
  EXPECT_EQ(package->weight_plan.coverage.external, 12);
  EXPECT_EQ(package->artifacts.size(), 5);

  auto session = artifacts::ModelLocator::OpenLocal(fixture);
  ASSERT_TRUE(session.ok()) << session.status();
  artifacts::ArtifactLimits limits;
  limits.max_active_mappings = 1;
  limits.max_mapped_bytes = kMappingLimit;
  limits.map_window_bytes = kMappingLimit;
  artifacts::ShardMappingPool pool(limits);
  std::set<uint32_t> mapped_parameters;
  std::set<std::string> mapped_shards;

  for (const auto& item : package->weight_plan.items) {
    ASSERT_TRUE(mapped_parameters.insert(item.parameter.value()).second)
        << "parameter mapped more than once: " << item.parameter.value();
    auto file = session->OpenRegular(item.source.shard);
    ASSERT_TRUE(file.ok()) << file.status();
    EXPECT_TRUE(item.source.expected_file.SameFileAndVersion(file->identity()));
    mapped_shards.insert(item.source.shard.string());
    {
      auto lease = pool.Map(*file, item.source.file_range);
      ASSERT_TRUE(lease.ok()) << lease.status();
      ASSERT_EQ(lease->bytes().size(), item.source.file_range.size);
      EXPECT_EQ(pool.active_regions(), 1);
      EXPECT_GT(pool.active_mapped_bytes(), 0);
      const std::byte expected = static_cast<std::byte>(item.parameter.value() + 1U);
      for (const std::byte value : lease->bytes()) EXPECT_EQ(value, expected);
    }
    EXPECT_EQ(pool.active_regions(), 0);
    EXPECT_EQ(pool.active_mapped_bytes(), 0);
  }

  EXPECT_EQ(mapped_parameters.size(), package->weight_plan.coverage.expected);
  EXPECT_EQ(mapped_shards, (std::set<std::string>{"model-00001-of-00002.safetensors",
                                                  "model-00002-of-00002.safetensors"}));
}

}  // namespace
}  // namespace inferx
