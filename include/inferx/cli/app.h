#ifndef INFERX_CLI_APP_H_
#define INFERX_CLI_APP_H_

#include "absl/status/statusor.h"

#include "inferx/command/options.h"

namespace inferx::cli {

using absl::StatusOr;

/// \brief Parses argv into an Invocation without executing anything.
///
/// \param argc The number of elements in argv.
/// \param argv The command line arguments, including the program name.
/// \return     A StatusOr containing the parsed Invocation, or an error if parsing failed.
StatusOr<command::Invocation> ParseFromCommandLine(int argc, const char* const* argv);

/// \brief Applies --log-level/--log-file to the process logger.
///
/// Called after parsing, before dispatch. Logs go to stderr unless
/// --log-file is given; an unopenable log file keeps stderr and is
/// reported through the logger itself.
void ConfigureLogging(const command::GlobalOptions& global);

}  // namespace inferx::cli

#endif  // INFERX_CLI_APP_H_
