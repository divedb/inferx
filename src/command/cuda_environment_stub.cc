#include <ostream>

#include "cuda_environment.h"

namespace inferx::command::internal {

void PrintCudaEnvironment(std::ostream& output) {
  output << "cuda.driver_version=not-compiled\n"
         << "cuda.runtime_version=not-compiled\n"
         << "cuda.device_count=0\n";
}

}  // namespace inferx::command::internal
