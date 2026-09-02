#include <exception>
#include <iostream>

#include "inferx/cli/app.h"
#include "inferx/command/dispatcher.h"

int main(int argc, char** argv) {
  try {
    auto dispatcher = inferx::command::CreateDefaultDispatcher(std::cout, std::cerr);
    return inferx::cli::Run(argc, argv, *dispatcher, std::cout, std::cerr);
  } catch (const std::exception& exception) {
    std::cerr << "error: inferx: unexpected exception: " << exception.what() << '\n';
  } catch (...) {
    std::cerr << "error: inferx: unexpected non-standard exception\n";
  }
  return static_cast<int>(inferx::command::ExitCode::kRuntime);
}
