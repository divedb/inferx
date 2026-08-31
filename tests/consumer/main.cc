// Downstream consumer entry point: proves the installed package, headers, and
// static library work with only CMAKE_PREFIX_PATH pointing at the prefix.
#include <cstdlib>
#include <iostream>
#include <string>

#include "inferx/base/version.h"

int main() {
  const char* expected_env = std::getenv("INFERX_EXPECTED_VERSION");
  if (expected_env == nullptr) {
    std::cerr << "INFERX_EXPECTED_VERSION not set by the driver\n";
    return 1;
  }
  const std::string expected = expected_env;
  const std::string actual(inferx::GetVersionString());
  if (actual != expected) {
    std::cerr << "installed version " << actual << " != expected " << expected << "\n";
    return 1;
  }
  const inferx::Version version = inferx::GetVersion();
  std::cout << "consumer ok: " << version.major << "." << version.minor << "." << version.patch
            << "\n";
  return 0;
}
