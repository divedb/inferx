#include "cuda_environment.h"

#include <cuda_runtime_api.h>

#include <ostream>
#include <string>

namespace inferx::command::internal {

void PrintCudaEnvironment(std::ostream& output) {
  int driver_version = 0;
  int runtime_version = 0;
  int device_count = 0;
  const cudaError_t driver_status = cudaDriverGetVersion(&driver_version);
  const cudaError_t runtime_status = cudaRuntimeGetVersion(&runtime_version);
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  output << "cuda.driver_version="
         << (driver_status == cudaSuccess ? std::to_string(driver_version) : "unavailable") << '\n'
         << "cuda.runtime_version="
         << (runtime_status == cudaSuccess ? std::to_string(runtime_version) : "unavailable")
         << '\n'
         << "cuda.device_count=" << (count_status == cudaSuccess ? device_count : 0) << '\n';
  if (count_status != cudaSuccess) return;

  for (int index = 0; index < device_count; ++index) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, index) != cudaSuccess) continue;
    output << "cuda.device." << index << ".name=" << properties.name << '\n'
           << "cuda.device." << index << ".compute_capability=" << properties.major << '.'
           << properties.minor << '\n'
           << "cuda.device." << index << ".memory_bytes=" << properties.totalGlobalMem << '\n';
  }
}

}  // namespace inferx::command::internal
