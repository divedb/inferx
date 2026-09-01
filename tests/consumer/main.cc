// Downstream consumer entry point: proves the installed package, headers, and
// static library work with only CMAKE_PREFIX_PATH pointing at the prefix.
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"
#include "inferx/base/version.h"
#include "inferx/config/parsed_config.h"
#include "inferx/engine/execution_ticket.h"
#include "inferx/ops/kernel_key.h"
#include "inferx/runtime/buffer_pool.h"
#include "inferx/scheduler/work_kind.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/shape.h"

int main() {
  const char* expected_env = std::getenv("INFERX_EXPECTED_VERSION");
  if (expected_env == nullptr) {
    std::cerr << "INFERX_EXPECTED_VERSION not set by the driver\n";
    return 1;
  }
  const std::string expected = expected_env;
  const std::string actual(inferx::GetVersionString());
  if (actual != expected) {
    std::cerr << "installed version " << actual << " != expected " << expected << "\n";
    return 1;
  }
  const inferx::RequestId request(1);
  const absl::StatusOr<inferx::ErrorReason> reason = inferx::GetErrorReason(absl::OkStatus());
  if (!reason.ok() || *reason != inferx::ErrorReason::kNone) {
    std::cerr << "installed status conventions did not behave\n";
    return 1;
  }
  const inferx::TokenCount tokens = inferx::TokenCount::FromUint64(2, "consumer").value();
  if (request.value() != 1 || tokens.value() != 2) {
    std::cerr << "installed value types did not behave\n";
    return 1;
  }
  const inferx::config::ParsedConfig config;
  const inferx::ExecutionTicket ticket{inferx::ExecutionTicketId(1), inferx::StepId(1), 1};
  if (config.MaxActiveSequences.value != 256 || ticket.item_count != 1 ||
      inferx::ToString(inferx::WorkKind::kPrefill) != "prefill") {
    std::cerr << "installed M1 module contracts did not behave\n";
    return 1;
  }
  const inferx::Version version = inferx::GetVersion();
  const inferx::Shape scalar = inferx::Shape::Create({}).value();
  inferx::CpuAllocator allocator;
  inferx::Buffer backing =
      allocator
          .Allocate(inferx::AllocationRequest{inferx::Device::Host(), inferx::MemoryKind::kHost,
                                              inferx::ByteCount(64), inferx::ByteCount(64),
                                              inferx::MemoryCategory::kTest})
          .value();
  inferx::FixedBufferPool pool =
      inferx::FixedBufferPool::Create(
          std::move(backing), inferx::PoolGeometry{1, inferx::ByteCount(64), inferx::ByteCount(64),
                                                   inferx::PoolGeneration(0)})
          .value();
  inferx::BufferLease lease = pool.Acquire().value();
  inferx::ops::KernelKey kernel_key;
  kernel_key.rank = 1;
  kernel_key.dimensions[0] = 1;
  if (scalar.NumElements().value() != 1 || inferx::ops::StableKernelKeyHash(kernel_key) == 0 ||
      !lease.Release().ok() || !pool.Close().ok()) {
    std::cerr << "installed tensor/runtime contracts did not behave\n";
    return 1;
  }
  std::cout << "consumer ok: " << version.major << "." << version.minor << "." << version.patch
            << "\n";
  return 0;
}
