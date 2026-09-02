# Dedicated installed-dependency consumer fixture (ADR 0002): proves the
# installed-dependency strategy — the pinned Abseil
# installs into its own prefix and is consumed only via find_package(absl
# CONFIG). This is the pattern InferXConfig.cmake.in uses for public
# absl::Status exposure.
cmake_minimum_required(VERSION 3.28)
project(inferx_absl_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(absl CONFIG REQUIRED)

add_executable(inferx_absl_consumer main.cc)
target_link_libraries(inferx_absl_consumer PRIVATE
  absl::status absl::statusor absl::strings)
