#include <benchmark/benchmark.h>

#include <cstdint>
#include <filesystem>

#include "inferx/artifacts/mapped_region.h"
#include "inferx/artifacts/model_locator.h"
#include "inferx/artifacts/safe_relative_path.h"
#include "inferx/artifacts/safetensors_reader.h"
#include "inferx/model/model_package.h"

namespace inferx {
namespace {

const std::filesystem::path& FixturePath() {
  static const std::filesystem::path path =
      std::filesystem::path(INFERX_ARTIFACT_FIXTURE_DIR) / "tiny_llama";
  return path;
}

void ArtifactInspect(benchmark::State& state) {
  model::ModelArtifactLoader loader;
  for (auto _ : state) {
    static_cast<void>(_);
    auto package = loader.Inspect(FixturePath());
    if (!package.ok()) {
      state.SkipWithError(package.status().ToString());
      return;
    }
    benchmark::DoNotOptimize(package->weight_plan.items.data());
  }
  state.SetItemsProcessed(state.iterations());
}

void SafeTensorHeaderParse(benchmark::State& state) {
  auto session = artifacts::ModelLocator::OpenLocal(FixturePath());
  if (!session.ok()) {
    state.SkipWithError(session.status().ToString());
    return;
  }
  auto path = artifacts::SafeRelativePath::Parse("model-00001-of-00002.safetensors");
  if (!path.ok()) {
    state.SkipWithError(path.status().ToString());
    return;
  }
  auto file = session->OpenRegular(*path);
  if (!file.ok()) {
    state.SkipWithError(file.status().ToString());
    return;
  }
  artifacts::SafeTensorReader reader;
  for (auto _ : state) {
    static_cast<void>(_);
    auto metadata = reader.ReadHeader(*file);
    if (!metadata.ok()) {
      state.SkipWithError(metadata.status().ToString());
      return;
    }
    benchmark::DoNotOptimize(metadata->tensors.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(file->identity().size));
}

void PlannedTensorMap(benchmark::State& state) {
  constexpr uint64_t kMappingLimit = 64ULL * 1024;
  model::ModelArtifactLoader loader;
  auto package = loader.Inspect(FixturePath());
  if (!package.ok()) {
    state.SkipWithError(package.status().ToString());
    return;
  }
  auto session = artifacts::ModelLocator::OpenLocal(FixturePath());
  if (!session.ok()) {
    state.SkipWithError(session.status().ToString());
    return;
  }
  artifacts::ArtifactLimits limits;
  limits.max_active_mappings = 1;
  limits.max_mapped_bytes = kMappingLimit;
  limits.map_window_bytes = kMappingLimit;
  artifacts::ShardMappingPool pool(limits);
  uint64_t bytes_per_iteration = 0;
  for (const auto& item : package->weight_plan.items) {
    bytes_per_iteration += item.source.file_range.size;
  }

  for (auto _ : state) {
    static_cast<void>(_);
    for (const auto& item : package->weight_plan.items) {
      auto file = session->OpenRegular(item.source.shard);
      if (!file.ok()) {
        state.SkipWithError(file.status().ToString());
        return;
      }
      auto lease = pool.Map(*file, item.source.file_range);
      if (!lease.ok()) {
        state.SkipWithError(lease.status().ToString());
        return;
      }
      benchmark::DoNotOptimize(lease->bytes().data());
    }
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(package->weight_plan.items.size()));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes_per_iteration));
}

BENCHMARK(ArtifactInspect);
BENCHMARK(SafeTensorHeaderParse);
BENCHMARK(PlannedTensorMap);

}  // namespace
}  // namespace inferx

BENCHMARK_MAIN();
