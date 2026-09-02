#ifndef INFERX_SRC_COMMAND_CUDA_ENVIRONMENT_H_
#define INFERX_SRC_COMMAND_CUDA_ENVIRONMENT_H_

#include <iosfwd>

namespace inferx::command::internal {

// Prints stable key/value diagnostics without exposing CUDA SDK types to the
// hardware-neutral command layer.
void PrintCudaEnvironment(std::ostream& output);

}  // namespace inferx::command::internal

#endif  // INFERX_SRC_COMMAND_CUDA_ENVIRONMENT_H_
