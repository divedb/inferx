// inferx-info: build/toolchain metadata smoke CLI.
//
// Proves target linkage and supplies diagnostics. Deliberately hand-rolled
// argument handling — no general JSON or argument-parsing dependency.

#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "inferx/base/version.h"

namespace {

// Filled by compile definitions from CMake; values carry no paths or
// machine-specific data.
constexpr std::string_view kCompilerId = INFERX_INFO_CXX_COMPILER_ID;
constexpr std::string_view kCompilerVersion = INFERX_INFO_CXX_COMPILER_VERSION;
constexpr std::string_view kCxxStandard = INFERX_INFO_CXX_STANDARD;
constexpr std::string_view kBuildType = INFERX_INFO_BUILD_TYPE;
constexpr std::string_view kFeatures = INFERX_INFO_FEATURES;
#ifdef INFERX_INFO_CUDA_ARCHITECTURES
constexpr std::string_view kCudaArchitectures = INFERX_INFO_CUDA_ARCHITECTURES;
#endif

void PrintVersion() {
  // Stable single-line output; CI matches on it exactly.
  const std::string_view version = inferx::GetVersionString();
  std::printf("inferx-info %.*s\n", static_cast<int>(version.size()), version.data());
}

void PrintBuild() {
  const inferx::Version version = inferx::GetVersion();
  std::printf("inferx: %d.%d.%d\n", version.major, version.minor, version.patch);
  std::printf("compiler: %.*s %.*s\n", static_cast<int>(kCompilerId.size()), kCompilerId.data(),
              static_cast<int>(kCompilerVersion.size()), kCompilerVersion.data());
  std::printf("c++-standard: %.*s\n", static_cast<int>(kCxxStandard.size()), kCxxStandard.data());
  std::printf("build-type: %.*s\n", static_cast<int>(kBuildType.size()), kBuildType.data());
  std::printf("features: %.*s\n", static_cast<int>(kFeatures.size()), kFeatures.data());
#ifdef INFERX_INFO_CUDA_ARCHITECTURES
  std::printf("cuda-architectures: %.*s\n", static_cast<int>(kCudaArchitectures.size()),
              kCudaArchitectures.data());
#endif
}

void PrintUsage(std::FILE* out) {
  std::fprintf(out,
               "usage: inferx-info [--version | --build | --help]\n"
               "  --version  print the single-line InferX version\n"
               "  --build    print build/toolchain metadata\n"
               "  --help     print this message\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    PrintUsage(stdout);
    return 0;
  }
  if (argc != 2) {
    std::fprintf(stderr, "inferx-info: exactly one option expected\n");
    PrintUsage(stderr);
    return 2;
  }
  const std::string_view argument = argv[1];
  if (argument == "--version") {
    PrintVersion();
    return 0;
  }
  if (argument == "--build") {
    PrintBuild();
    return 0;
  }
  if (argument == "--help") {
    PrintUsage(stdout);
    return 0;
  }
  std::fprintf(stderr, "inferx-info: unknown option '%.*s'\n", static_cast<int>(argument.size()),
               argument.data());
  PrintUsage(stderr);
  return 2;
}
