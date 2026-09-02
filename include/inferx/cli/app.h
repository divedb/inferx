#ifndef INFERX_CLI_APP_H_
#define INFERX_CLI_APP_H_

#include <ostream>

#include "inferx/command/dispatcher.h"

namespace inferx::cli {

int Run(int argc, const char* const* argv, command::Dispatcher& dispatcher, std::ostream& output,
        std::ostream& error);

}  // namespace inferx::cli

#endif  // INFERX_CLI_APP_H_
