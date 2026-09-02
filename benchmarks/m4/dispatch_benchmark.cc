#include <cstdint>
#include <utility>

#include "benchmark/benchmark.h"
#include "inferx/ops/backend_capability.h"
#include "inferx/ops/kernel_registry.h"

namespace inferx::ops {
namespace {

void FrozenRegistryLookup(benchmark::State& state) {
  BackendCapability capability;
  capability.backend = BackendId::kReference;
  capability.capability_id = "benchmark.reference.gemm";
  capability.op = OpKind::kGemm;
  capability.device_kind = DeviceKind::kHost;
  capability.input_dtype_mask = DTypeMask(DType::kFloat32);
  capability.weight_dtype_mask = DTypeMask(DType::kFloat32);
  capability.output_dtype_mask = DTypeMask(DType::kFloat32);
  capability.rank = 3;
  capability.dimensions[0] = {0, 512};
  capability.dimensions[1] = {1, 4096};
  capability.dimensions[2] = {1, 4096};
  capability.minimum_operand_alignment = 4;
  KernelRegistry registry(1);
  if (!registry.Register(std::move(capability)).ok() || !registry.Freeze().ok()) {
    state.SkipWithError("failed to construct benchmark registry");
    return;
  }
  KernelKey key;
  key.op = OpKind::kGemm;
  key.rank = 3;
  key.dimensions[0] = 1;
  key.dimensions[1] = 4096;
  key.dimensions[2] = 4096;
  key.alignment_class = 4;
  for (auto _ : state) {
    auto selection = registry.Select(key);
    if (!selection.ok()) {
      state.SkipWithError("lookup failed");
      break;
    }
    benchmark::DoNotOptimize(selection->registry_index);
  }
}

BENCHMARK(FrozenRegistryLookup);

}  // namespace
}  // namespace inferx::ops

BENCHMARK_MAIN();
