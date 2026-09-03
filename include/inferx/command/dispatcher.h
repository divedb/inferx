#ifndef INFERX_COMMAND_DISPATCHER_H_
#define INFERX_COMMAND_DISPATCHER_H_

#include <iostream>
#include <memory>

#include "inferx/command/options.h"

namespace inferx::command {

void Dispatch(const Invocation& invocation);

}  // namespace inferx::command

#endif  // INFERX_COMMAND_DISPATCHER_H_
