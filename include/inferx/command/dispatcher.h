#ifndef INFERX_COMMAND_DISPATCHER_H_
#define INFERX_COMMAND_DISPATCHER_H_

#include <iostream>
#include <memory>

#include "inferx/command/options.h"

namespace inferx::command {

/// \brief Process exit status of one command execution.
enum class ExitCode : int {
  kSuccess = 0,
  kUsage = 2,
  kModel = 3,
  kRuntime = 4,
  kBenchmark = 5,
  kUnavailable = 6,
};

// Executes one parsed command line. Diagnostics are emitted through the
// process logger (inferx/base/log.h: stderr or --log-file); `results`
// carries command output (stdout by default, injectable for tests).
class Dispatcher {
 public:
  virtual ~Dispatcher() = default;

  virtual ExitCode Dispatch(const Invocation& invocation) = 0;
};

std::unique_ptr<Dispatcher> CreateDefaultDispatcher(std::ostream& results = std::cout);

}  // namespace inferx::command

#endif  // INFERX_COMMAND_DISPATCHER_H_
