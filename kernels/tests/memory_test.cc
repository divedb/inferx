#include "inferx/kernels/memory.h"

#include <bit>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

namespace {

using inferx::kernels::Backend;
using inferx::kernels::CopyAsync;
using inferx::kernels::DeviceBytes;
using inferx::kernels::ExecutionContext;
using inferx::kernels::FillAsync;
using inferx::kernels::FillBytesAsync;
using inferx::kernels::FillPattern;
using inferx::kernels::IsBackendCompiled;
using inferx::kernels::MutableDeviceBytes;
using inferx::kernels::Status;
using inferx::kernels::StatusCode;
using inferx::kernels::ZeroAsync;

#define CHECK(condition)                                                      \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: "         \
                << #condition << '\n';                                        \
      return false;                                                           \
    }                                                                         \
  } while (false)

constexpr ExecutionContext kCpuContext{Backend::kCPU, {}};

bool TestFillPatterns() {
  constexpr auto byte = FillPattern::From<uint8_t>(0xA5);
  constexpr auto half = FillPattern::FromStorageBits16(0x3C00);
  constexpr auto word = FillPattern::From<uint32_t>(0xDEADBEEF);
  constexpr auto wide = FillPattern::From<uint64_t>(0x0123456789ABCDEFULL);
  constexpr auto fp32 = FillPattern::From<float>(-3.25F);

  CHECK(byte.Width() == 1);
  CHECK(byte.Bits() == 0xA5);
  CHECK(half.Width() == 2);
  CHECK(half.Bits() == 0x3C00);
  CHECK(word.Width() == 4);
  CHECK(word.Bits() == 0xDEADBEEF);
  CHECK(wide.Width() == 8);
  CHECK(wide.Bits() == 0x0123456789ABCDEFULL);
  CHECK(fp32.Bits() == std::bit_cast<uint32_t>(-3.25F));
  return true;
}

bool TestStatus() {
  const std::string long_operation(80, 'o');
  const std::string long_message(180, 'm');
  const Status status =
      Status::BackendError(long_operation, long_message, -123);
  CHECK(!status.IsOk());
  CHECK(status.Code() == StatusCode::kBackendError);
  CHECK(status.Operation().size() == Status::kOperationCapacity);
  CHECK(status.Message().size() == Status::kMessageCapacity);
  CHECK(status.NativeCode() == -123);
  CHECK(Status{}.IsOk());
  return true;
}

bool TestBackendConfiguration() {
  CHECK(IsBackendCompiled(Backend::kCPU));
  CHECK(IsBackendCompiled(Backend::kCuda) == inferx::kernels::kHasCuda);
  CHECK(IsBackendCompiled(Backend::kRocm) == inferx::kernels::kHasRocm);
  CHECK(IsBackendCompiled(Backend::kAscend) == inferx::kernels::kHasAscend);
  return true;
}

bool TestEmptyAndAliasNoOps() {
  for (const Backend backend : {Backend::kCuda, Backend::kRocm,
                                Backend::kAscend, Backend::kCPU}) {
    const ExecutionContext context{backend, {}};
    CHECK(ZeroAsync({}, context).IsOk());
    CHECK(FillBytesAsync({}, 0x4A, context).IsOk());
    CHECK(FillAsync({}, FillPattern::From<uint32_t>(7), context).IsOk());
    CHECK(CopyAsync({}, {}, context).IsOk());

    void* pointer = reinterpret_cast<void*>(uintptr_t{0x1000});
    CHECK(CopyAsync({pointer, 64}, {pointer, 64}, context).IsOk());
  }
  return true;
}

bool TestValidation() {
  const auto IsInvalid = [](const Status& status,
                            std::string_view operation) {
    return status.Code() == StatusCode::kInvalidArgument &&
           status.Operation() == operation;
  };

  CHECK(IsInvalid(ZeroAsync({nullptr, 1}, kCpuContext), "memory.zero"));
  CHECK(IsInvalid(FillBytesAsync({nullptr, 1}, 1, kCpuContext),
                  "memory.fill_bytes"));
  CHECK(IsInvalid(FillAsync({nullptr, 4}, FillPattern::From<uint32_t>(1),
                            kCpuContext),
                  "memory.fill"));
  CHECK(IsInvalid(FillAsync(
                      {reinterpret_cast<void*>(uintptr_t{0x1000}), 4}, {},
                      kCpuContext),
                  "memory.fill"));
  CHECK(IsInvalid(FillAsync(
                      {reinterpret_cast<void*>(uintptr_t{0x1000}), 3},
                      FillPattern::From<uint16_t>(1), kCpuContext),
                  "memory.fill"));
  CHECK(IsInvalid(FillAsync(
                      {reinterpret_cast<void*>(uintptr_t{0x1001}), 4},
                      FillPattern::From<uint32_t>(1), kCpuContext),
                  "memory.fill"));
  CHECK(IsInvalid(CopyAsync({nullptr, 4}, {nullptr, 8}, kCpuContext),
                  "memory.copy"));
  CHECK(IsInvalid(CopyAsync({nullptr, 4},
                            {reinterpret_cast<void*>(uintptr_t{0x2000}), 4},
                            kCpuContext),
                  "memory.copy"));
  CHECK(IsInvalid(CopyAsync(
                      {reinterpret_cast<void*>(uintptr_t{0x1000}), 16},
                      {reinterpret_cast<void*>(uintptr_t{0x1008}), 16},
                      kCpuContext),
                  "memory.copy"));
  CHECK(IsInvalid(CopyAsync(
                      {reinterpret_cast<void*>(uintptr_t{0x1008}), 16},
                      {reinterpret_cast<void*>(uintptr_t{0x1000}), 16},
                      kCpuContext),
                  "memory.copy"));
  return true;
}

bool TestUnavailableAndUnsupported() {
  void* first = reinterpret_cast<void*>(uintptr_t{0x1000});
  void* second = reinterpret_cast<void*>(uintptr_t{0x2000});

  CHECK(ZeroAsync({first, 16}, kCpuContext).Code() ==
        StatusCode::kUnsupported);
  CHECK(FillBytesAsync({first, 16}, 1, kCpuContext).Code() ==
        StatusCode::kUnsupported);
  CHECK(FillAsync({first, 16}, FillPattern::From<uint32_t>(7), kCpuContext)
            .Code() == StatusCode::kUnsupported);
  CHECK(CopyAsync({first, 16}, {second, 16}, kCpuContext).Code() ==
        StatusCode::kUnsupported);

  if (!inferx::kernels::kHasCuda) {
    const ExecutionContext cuda{Backend::kCuda, {}};
    CHECK(ZeroAsync({first, 16}, cuda).Code() == StatusCode::kUnavailable);
  }
  if (!inferx::kernels::kHasRocm) {
    const ExecutionContext rocm{Backend::kRocm, {}};
    CHECK(ZeroAsync({first, 16}, rocm).Code() == StatusCode::kUnavailable);
  }
  if (!inferx::kernels::kHasAscend) {
    const ExecutionContext ascend{Backend::kAscend, {}};
    CHECK(ZeroAsync({first, 16}, ascend).Code() == StatusCode::kUnavailable);
  }
  return true;
}

}  // namespace

int main() {
  if (!TestFillPatterns() || !TestStatus() || !TestBackendConfiguration() ||
      !TestEmptyAndAliasNoOps() || !TestValidation() ||
      !TestUnavailableAndUnsupported()) {
    return 1;
  }
  return 0;
}
