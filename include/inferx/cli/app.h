#ifndef INFERX_CLI_APP_H_
#define INFERX_CLI_APP_H_

#include <iostream>
#include <ostream>

#include "inferx/command/options.h"

namespace inferx::cli {

/// \brief Outcome of command-line parsing.
///
/// `should_run` is false for exits that only produce terminal output
/// (help, the `--version` flag, parse and usage errors); `exit_code` is
/// then the process exit status and `invocation` is left at its default.
struct ParseResult {
  command::ExitCode exit_code = command::ExitCode::kSuccess;
  bool should_run = false;
  command::Invocation invocation;
};

/// \brief Parses argv into an Invocation without executing anything.
///
/// Help text is written to `output`, parse and usage errors to `error`
/// (both injectable for tests). Semantic validation (sampling ranges,
/// required flag combinations) happens here, so a `should_run` result is
/// ready to dispatch.
ParseResult ParseFromCommandLine(int argc, const char* const* argv,
                                 std::ostream& output = std::cout,
                                 std::ostream& error = std::cerr);

/// \brief Applies --log-level/--log-file to the process logger.
///
/// Called after parsing, before dispatch. Logs go to stderr unless
/// --log-file is given; an unopenable log file keeps stderr and is
/// reported through the logger itself.
void ConfigureLogging(const command::GlobalOptions& global);

}  // namespace inferx::cli

#endif  // INFERX_CLI_APP_H_
