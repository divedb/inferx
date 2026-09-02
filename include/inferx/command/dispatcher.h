#ifndef INFERX_COMMAND_DISPATCHER_H_
#define INFERX_COMMAND_DISPATCHER_H_

#include <memory>
#include <ostream>

#include "inferx/command/options.h"

namespace inferx::command {

enum class ExitCode : int {
  kSuccess = 0,
  kUsage = 2,
  kModel = 3,
  kRuntime = 4,
  kBenchmark = 5,
  kUnavailable = 6,
};

class Dispatcher {
 public:
  virtual ~Dispatcher() = default;

  virtual ExitCode Serve(const GlobalOptions& global, const ServeOptions& options) = 0;
  virtual ExitCode Bench(const GlobalOptions& global, const BenchmarkOptions& options) = 0;
  virtual ExitCode Run(const GlobalOptions& global, const RunOptions& options) = 0;
  virtual ExitCode Chat(const GlobalOptions& global, const ClientOptions& options) = 0;
  virtual ExitCode Complete(const GlobalOptions& global, const ClientOptions& options) = 0;
  virtual ExitCode Inspect(const GlobalOptions& global, const InspectOptions& options) = 0;
  virtual ExitCode Download(const GlobalOptions& global, const DownloadOptions& options) = 0;
  virtual ExitCode Simulate(const GlobalOptions& global, const SimulateOptions& options) = 0;
  virtual ExitCode Version(const GlobalOptions& global) = 0;
  virtual ExitCode Environment(const GlobalOptions& global) = 0;
};

std::unique_ptr<Dispatcher> CreateDefaultDispatcher(std::ostream& output, std::ostream& error);

}  // namespace inferx::command

#endif  // INFERX_COMMAND_DISPATCHER_H_
