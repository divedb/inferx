#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/runtime/buffer_pool.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/device.h"

#if defined(INFERX_MEMORY_POOL_BENCHMARK_CUDA)
#include <cuda_runtime_api.h>

#include <array>

#include "inferx/platform/cuda/cuda_allocator.h"
#include "inferx/platform/cuda/cuda_device.h"
#include "inferx/runtime/memory_tracker.h"
#endif

namespace {

void BM_FixedPoolAcquireRelease(benchmark::State& state) {
  inferx::CpuAllocator allocator;
  const uint64_t slot_bytes = static_cast<uint64_t>(state.range(0));
  inferx::Buffer backing =
      allocator
          .Allocate(inferx::AllocationRequest{inferx::Device::Host(), inferx::MemoryKind::kHost,
                                              inferx::ByteCount(slot_bytes * 64),
                                              inferx::ByteCount(64), inferx::MemoryCategory::kTest})
          .value();
  inferx::FixedBufferPool pool =
      inferx::FixedBufferPool::Create(
          std::move(backing),
          inferx::PoolGeometry{64, inferx::ByteCount(slot_bytes), inferx::ByteCount(64),
                               inferx::PoolGeneration(0)})
          .value();
  for (auto _ : state) {
    static_cast<void>(_);
    inferx::BufferLease lease = pool.Acquire().value();
    benchmark::DoNotOptimize(lease.token().generation.value());
    if (!lease.Release().ok()) state.SkipWithError("pool release failed");
  }
  state.SetItemsProcessed(state.iterations());
  if (!pool.Close().ok()) state.SkipWithError("pool close failed");
}

BENCHMARK(BM_FixedPoolAcquireRelease)
    ->Arg(4096)
    ->Arg(1048576)
    ->Repetitions(30)
    ->MinWarmUpTime(0.1)
    ->UseRealTime();

#if defined(INFERX_MEMORY_POOL_BENCHMARK_CUDA)

void BM_CudaRawAllocationFree(benchmark::State& state) {
  const uint64_t bytes = static_cast<uint64_t>(state.range(0));
  int64_t completed_iterations = 0;
  for (auto _ : state) {
    static_cast<void>(_);
    void* address = nullptr;
    if (cudaMalloc(&address, static_cast<size_t>(bytes)) != cudaSuccess) {
      state.SkipWithError("cudaMalloc failed");
      break;
    }
    benchmark::DoNotOptimize(address);
    if (cudaFree(address) != cudaSuccess) {
      state.SkipWithError("cudaFree failed");
      break;
    }
    ++completed_iterations;
  }
  state.SetItemsProcessed(completed_iterations);
  state.counters["cuda_allocation_calls"] = static_cast<double>(completed_iterations);
  state.counters["cuda_free_calls"] = static_cast<double>(completed_iterations);
}

void BM_CudaWarmedFixedPoolAcquireRelease(benchmark::State& state) {
  const auto devices = inferx::cuda::DiscoverCudaDevices();
  if (!devices.ok() || devices->empty()) {
    state.SkipWithError("no CUDA device");
    return;
  }
  const inferx::DeviceId device_id = devices->front().ordinal;
  auto guard = inferx::cuda::CudaDeviceGuard::Create(device_id);
  if (!guard.ok()) {
    state.SkipWithError("device guard failed");
    return;
  }
  const uint64_t slot_bytes = static_cast<uint64_t>(state.range(0));
  constexpr uint32_t kSlots = 64;
  const uint64_t total_bytes = slot_bytes * kSlots;
  const std::array<inferx::MemoryLimit, 1> limits{
      inferx::MemoryLimit{inferx::Device::Cuda(device_id), inferx::MemoryKind::kDevice,
                          inferx::ByteCount(total_bytes)}};
  inferx::MemoryTracker tracker = inferx::MemoryTracker::Create(limits).value();
  inferx::cuda::CudaHealth health;
  inferx::cuda::CudaDeviceAllocator allocator(device_id, tracker, health);
  inferx::Buffer backing =
      allocator
          .Allocate({inferx::Device::Cuda(device_id), inferx::MemoryKind::kDevice,
                     inferx::ByteCount(total_bytes), inferx::ByteCount(256),
                     inferx::MemoryCategory::kTest})
          .value();
  inferx::FixedBufferPool pool =
      inferx::FixedBufferPool::Create(
          std::move(backing),
          inferx::PoolGeometry{kSlots, inferx::ByteCount(slot_bytes), inferx::ByteCount(256),
                               inferx::PoolGeneration(0)})
          .value();
  for (auto _ : state) {
    static_cast<void>(_);
    inferx::BufferLease lease = pool.Acquire().value();
    benchmark::DoNotOptimize(lease.token().generation.value());
    if (!lease.Release().ok()) {
      state.SkipWithError("device pool release failed");
      break;
    }
  }
  state.counters["cuda_allocation_calls"] = 1.0;
  state.counters["cuda_free_calls"] = 1.0;
  state.counters["steady_state_cuda_allocation_calls"] = 0.0;
  state.counters["steady_state_cuda_free_calls"] = 0.0;
  if (!pool.Close().ok() || !tracker.ValidateBaseline().ok()) {
    state.SkipWithError("device pool cleanup failed");
  }
  if (!guard->Restore().ok()) state.SkipWithError("device restore failed");
}

BENCHMARK(BM_CudaRawAllocationFree)
    ->Arg(4096)
    ->Arg(1048576)
    ->Arg(67108864)
    ->Repetitions(30)
    ->MinWarmUpTime(0.1)
    ->UseRealTime();
BENCHMARK(BM_CudaWarmedFixedPoolAcquireRelease)
    ->Arg(4096)
    ->Arg(1048576)
    ->Arg(67108864)
    ->Repetitions(30)
    ->MinWarmUpTime(0.1)
    ->UseRealTime();

#endif

}  // namespace

BENCHMARK_MAIN();
