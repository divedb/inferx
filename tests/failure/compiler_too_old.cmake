# Fixture: unsupported/too-old compiler. Simulates a
# sub-floor compiler for the configure-time floor check in
# InferXProjectOptions.cmake; must fail before compiling project sources.
cmake_minimum_required(VERSION 3.28)
project(compiler_floor_fixture LANGUAGES CXX)

set(CMAKE_CXX_COMPILER_ID "GNU")
set(CMAKE_CXX_COMPILER_VERSION "12.3.0")
include("${INFERX_ROOT}/cmake/InferXProjectOptions.cmake")
