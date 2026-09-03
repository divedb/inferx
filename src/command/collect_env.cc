#include <cstdlib>
#include <iostream>

#include "cuda_environment.h"
#include "common.h"

namespace inferx::command {
namespace {

void PrintEnvironment(const char* name) {
  const char* value = std::getenv(name);
  internal::kOut << "environment." << name << '=' << (value == nullptr ? "<unset>" : value) << '\n';
}

}  // namespace

void CollectEnv() {
  internal::kOut << "compiled.cuda=" << (INFERX_COMMAND_ENABLE_CUDA ? "true" : "false") << '\n'
                 << "compiled.rocm=false\n"
                 << "rocm.driver_version=not-compiled\n"
                 << "rocm.runtime_version=not-compiled\n"
                 << "rocm.device_count=0\n";
  internal::PrintCudaEnvironment(internal::kOut);
  PrintEnvironment("CUDA_VISIBLE_DEVICES");
  PrintEnvironment("HIP_VISIBLE_DEVICES");
  PrintEnvironment("ROCR_VISIBLE_DEVICES");
}

}  // namespace inferx::command
