cmake_minimum_required(VERSION 3.28)

project(inferx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Pinned dependencies are git submodules (third_party/README.md); CMake never
# touches the network. Uninitialized submodules fall back to any installed
# package so a checkout without submodules can still configure.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/abseil-cpp/CMakeLists.txt")
  set(ABSL_PROPAGATE_CXX_STD ON)
  add_subdirectory(third_party/abseil-cpp)
else()
  find_package(absl CONFIG REQUIRED)
endif()

# Build-time accelerator-backend detection (CUDA implemented; ROCm/Ascend
# detection scaffolding). Generates inferx/kernels/build_config.h.
include(cmake/InferxBackends.cmake)
inferx_detect_backends()

include(CTest)
if(BUILD_TESTING AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
  add_subdirectory(third_party/googletest)
endif()

add_subdirectory(kernels)

# src/cli, src/command, and apps are not wired in yet: the in-flight refactor
# still references removed inferx::base/inferx::artifacts targets and headers.
