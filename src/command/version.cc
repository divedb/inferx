#include <iostream>

#include "inferx/base/version.h"
#include "common.h"

namespace inferx::command {

void Version() {
  const inferx::Version version = GetVersion();
  internal::kOut << "inferx: " << version.major << '.' << version.minor << '.' << version.patch
                 << '\n'
                 << "git-revision: " << INFERX_COMMAND_GIT_REVISION << '\n'
                 << "compiler: " << INFERX_COMMAND_CXX_COMPILER_ID << ' '
                 << INFERX_COMMAND_CXX_COMPILER_VERSION << '\n'
                 << "c++-standard: " << INFERX_COMMAND_CXX_STANDARD << '\n'
                 << "build-type: " << INFERX_COMMAND_BUILD_TYPE << '\n'
                 << "features: " << INFERX_COMMAND_FEATURES << '\n';
#if INFERX_COMMAND_ENABLE_CUDA
  internal::kOut << "cuda-architectures: " << INFERX_COMMAND_CUDA_ARCHITECTURES << '\n';
#endif
}

}  // namespace inferx::command
