#include <exception>

#include "inferx/base/log.h"
#include "inferx/cli/app.h"
#include "inferx/command/dispatcher.h"

int main(int argc, char** argv) {
  try {
    const inferx::cli::ParseResult parsed = inferx::cli::ParseFromCommandLine(argc, argv);
    if (!parsed.should_run) return static_cast<int>(parsed.exit_code);

    inferx::cli::ConfigureLogging(parsed.invocation.global);
    auto dispatcher = inferx::command::CreateDefaultDispatcher();
    return static_cast<int>(dispatcher->Dispatch(parsed.invocation));
  } catch (const std::exception& exception) {
    LOG(ERROR) << "inferx: unexpected exception: " << exception.what();
  } catch (...) {
    LOG(ERROR) << "inferx: unexpected non-standard exception";
  }

  return static_cast<int>(inferx::command::ExitCode::kRuntime);
}
