// CUDA smoke host driver (m0.md section 10.2): device query, tiny device
// allocation, kernel launch, copy-back, full validation, checked errors, and
// release on every exit path. Not the M2 runtime design; private to tests.
#include "smoke.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "smoke_kernel.cuh"

namespace inferx::cuda_smoke {
namespace {

constexpr int kElementCount = 64;

bool AcceptedArchitectureContains(int major, int minor, std::string_view list) {
  // The CMake list uses semicolons ("89;90"); entries are two-digit SM ids.
  // "89" matches (8,9). Entries may also be "80a"/"90a" real/virtual forms;
  // only the plain digits are accepted for the runtime check.
  std::string wanted = std::to_string(major) + std::to_string(minor);
  size_t position = 0;
  while (position < list.size()) {
    const size_t end = list.find(';', position);
    std::string_view entry = list.substr(
        position, end == std::string_view::npos ? std::string_view::npos : end - position);
    if (entry == wanted) return true;
    if (end == std::string_view::npos) break;
    position = end + 1;
  }
  return false;
}

void ReportError(const char* call, cudaError_t error) {
  std::fprintf(stderr, "cuda smoke: %s failed: %s (%s)\n", call, cudaGetErrorString(error),
               cudaGetErrorName(error));
}

}  // namespace

SmokeResult RunSmoke(const char* accepted_architectures) {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess || device_count <= 0) {
    std::printf("cuda smoke: no visible CUDA device (%s); skipping\n", cudaGetErrorString(error));
    return SmokeResult::kNoDevice;
  }

  // Device 0 only, selected within this process for the duration of the run.
  error = cudaSetDevice(0);
  if (error != cudaSuccess) {
    ReportError("cudaSetDevice", error);
    return SmokeResult::kFailed;
  }

  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, 0);
  if (error != cudaSuccess) {
    ReportError("cudaGetDeviceProperties", error);
    return SmokeResult::kFailed;
  }

  std::printf("cuda smoke: device 0: %s, compute capability %d.%d, %.0f MiB\n", properties.name,
              properties.major, properties.minor,
              static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0));
  std::fflush(stdout);

  if (!AcceptedArchitectureContains(properties.major, properties.minor, accepted_architectures)) {
    std::fprintf(stderr,
                 "cuda smoke: compute capability %d.%d is outside the accepted "
                 "list '%s'; the support matrix must match the runner\n",
                 properties.major, properties.minor, accepted_architectures);
    return SmokeResult::kFailed;
  }

  int* device_buffer = nullptr;
  error = cudaMalloc(&device_buffer, kElementCount * sizeof(int));
  if (error != cudaSuccess) {
    ReportError("cudaMalloc", error);
    return SmokeResult::kFailed;
  }

  launch_inferx_smoke_write_pattern(device_buffer, kElementCount);
  error = cudaGetLastError();  // post-launch error check
  if (error != cudaSuccess) {
    ReportError("kernel launch", error);
    cudaFree(device_buffer);
    return SmokeResult::kFailed;
  }

  std::vector<int> host_buffer(kElementCount, 0);
  error = cudaMemcpy(host_buffer.data(), device_buffer, kElementCount * sizeof(int),
                     cudaMemcpyDeviceToHost);
  const cudaError_t copy_error = error;  // free regardless below
  error = cudaFree(device_buffer);
  if (copy_error != cudaSuccess) {
    ReportError("cudaMemcpy D2H", copy_error);
    return SmokeResult::kFailed;
  }
  if (error != cudaSuccess) {
    ReportError("cudaFree", error);
    return SmokeResult::kFailed;
  }

  for (int index = 0; index < kElementCount; ++index) {
    const int expected = index * 7 + 13;
    if (host_buffer[static_cast<size_t>(index)] != expected) {
      std::fprintf(stderr,
                   "cuda smoke: element %d is %d, expected %d — device-written "
                   "data did not validate\n",
                   index, host_buffer[static_cast<size_t>(index)], expected);
      return SmokeResult::kFailed;
    }
  }

  std::printf("cuda smoke: %d/%d elements validated\n", kElementCount, kElementCount);
  return SmokeResult::kOk;
}

}  // namespace inferx::cuda_smoke
