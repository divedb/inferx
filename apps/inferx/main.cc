#include <cstdlib>
#include <exception>

#include "absl/status/status.h"
#include "inferx/base/log.h"
#include "inferx/cli/app.h"
#include "inferx/command/dispatcher.h"

namespace {

// Parse-exit contract (see src/cli/app.cc): help and --version output is
// already printed and exits 0 (kCancelled); parse and usage errors exit
// with their status code (kUnknown == 2).
int ExitStatus(const absl::Status& status) {
  return status.code() == absl::StatusCode::kCancelled ? EXIT_SUCCESS
                                                       : static_cast<int>(status.code());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    inferx::log::Initialize();
    auto parsed = inferx::cli::ParseFromCommandLine(argc, argv);

    if (!parsed.ok()) return ExitStatus(parsed.status());

    inferx::cli::ConfigureLogging(parsed->global);
    inferx::command::Dispatch(*parsed);

    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    LOG(ERROR) << "inferx: unexpected exception: " << exception.what();
  } catch (...) {
    LOG(ERROR) << "inferx: unexpected non-standard exception";
  }

  return EXIT_FAILURE;
}
