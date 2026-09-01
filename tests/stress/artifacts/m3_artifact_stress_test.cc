#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/artifacts/mapped_region.h"
#include "inferx/artifacts/model_locator.h"
#include "inferx/artifacts/safe_relative_path.h"
#include "inferx/artifacts/safetensors_reader.h"
#include "inferx/model/model_package.h"

namespace inferx {
namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    std::vector<char> pattern{'/', 't', 'm', 'p', '/', 'i', 'n', 'f', 'e', 'r',
                              'x', '-', 'm', '3', '-', 's', 't', 'r', 'e', 's',
                              's', '-', 'X', 'X', 'X', 'X', 'X', 'X', '\0'};
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

std::vector<std::byte> ReadBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  const std::vector<char> characters{std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>()};
  std::vector<std::byte> bytes;
  bytes.reserve(characters.size());
  for (const char character : characters) {
    bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
  }
  return bytes;
}

void WriteBytes(const std::filesystem::path& path, std::span<const std::byte> bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

uint64_t NextRandom(uint64_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  return *state;
}

TEST(ArtifactStressTest, RepeatedlyOpensMapsAndReleasesPlannedTensorRanges) {
  constexpr uint64_t kOperations = 10'000;
  constexpr uint64_t kMappingLimit = 64ULL * 1024;
  const std::filesystem::path fixture = std::filesystem::path(INFERX_M3_FIXTURE_DIR) / "tiny_llama";
  model::ModelArtifactLoader loader;
  auto package = loader.Inspect(fixture);
  ASSERT_TRUE(package.ok()) << package.status();
  auto session = artifacts::ModelLocator::OpenLocal(fixture);
  ASSERT_TRUE(session.ok()) << session.status();
  artifacts::ArtifactLimits limits;
  limits.max_active_mappings = 1;
  limits.max_mapped_bytes = kMappingLimit;
  limits.map_window_bytes = kMappingLimit;
  artifacts::ShardMappingPool pool(limits);

  for (uint64_t operation = 0; operation < kOperations; ++operation) {
    const auto& item = package->weight_plan.items[static_cast<size_t>(
        operation % static_cast<uint64_t>(package->weight_plan.items.size()))];
    auto file = session->OpenRegular(item.source.shard);
    ASSERT_TRUE(file.ok()) << "operation " << operation << ": " << file.status();
    {
      auto lease = pool.Map(*file, item.source.file_range);
      ASSERT_TRUE(lease.ok()) << "operation " << operation << ": " << lease.status();
      ASSERT_EQ(lease->bytes().size(), item.source.file_range.size);
    }
    ASSERT_EQ(pool.active_regions(), 0) << "operation " << operation;
    ASSERT_EQ(pool.active_mapped_bytes(), 0) << "operation " << operation;
  }
}

TEST(ArtifactStressTest, DeterministicallyMutatesParserSeedWithoutCrashOrLeak) {
  constexpr uint64_t kOperations = 10'000;
  const std::filesystem::path corpus = std::filesystem::path(INFERX_M3_FIXTURE_DIR) / "safetensors";
  const std::vector<std::byte> seed = ReadBytes(corpus / "valid_scalar.safetensors");
  ASSERT_GT(seed.size(), 16);
  TemporaryDirectory temporary;
  auto session = artifacts::ModelLocator::OpenLocal(temporary.path());
  ASSERT_TRUE(session.ok()) << session.status();
  auto relative = artifacts::SafeRelativePath::Parse("mutated.safetensors");
  ASSERT_TRUE(relative.ok()) << relative.status();
  artifacts::SafeTensorReader reader;
  artifacts::ArtifactLimits limits;
  limits.max_safetensors_header_bytes = 4096;
  limits.max_json_bytes = 4096;
  uint64_t random = 0x4D33534146455459ULL;

  for (uint64_t operation = 0; operation < kOperations; ++operation) {
    std::vector<std::byte> mutated = seed;
    const uint64_t choice = NextRandom(&random) % 4;
    if (choice == 0) {
      mutated[static_cast<size_t>(NextRandom(&random) % 8)] ^= std::byte{0x80};
    } else if (choice == 1) {
      const size_t keep = static_cast<size_t>(NextRandom(&random) % mutated.size());
      mutated.resize(keep);
    } else if (choice == 2) {
      const size_t index = 8 + static_cast<size_t>(NextRandom(&random) % (mutated.size() - 8));
      mutated[index] ^= static_cast<std::byte>(1U << (NextRandom(&random) % 8));
    } else {
      mutated.push_back(static_cast<std::byte>(NextRandom(&random) & 0xFFU));
    }
    WriteBytes(temporary.path() / relative->string(), mutated);
    auto file = session->OpenRegular(*relative);
    ASSERT_TRUE(file.ok()) << "operation " << operation << ": " << file.status();
    static_cast<void>(reader.ReadHeader(*file, limits));
  }
}

}  // namespace
}  // namespace inferx
